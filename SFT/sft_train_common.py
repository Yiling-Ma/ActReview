"""
Common full-SFT training utilities for conversational data.

This module centralizes the training pipeline and exposes a task-specific CLI
entry helper so task1/task2 can run as independent scripts with separate models.
"""

import argparse
import importlib.util
import json
import math
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from torch.distributed.elastic.multiprocessing.errors import record
from accelerate.utils import set_seed
from tqdm import tqdm


DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B-Base"
MAX_SEQ_LEN = 4096
IGNORE_INDEX = -100


class ReviewSFTDataset(Dataset):
    """
    Loads SFT records and tokenizes using the chat template.
    Supports multi-turn conversations: loss is computed on all assistant turns.
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.records = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        print(f"Loaded {len(self.records)} records from {data_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self._tokenize(self.records[idx]["messages"])

    def _apply_template(self, messages: list[dict], add_generation_prompt: bool) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )["input_ids"][0]

    def _tokenize(self, messages: list[dict]) -> dict:
        full_text = self._apply_template(messages, add_generation_prompt=False)
        input_ids = self._encode(full_text)
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            prefix_ids = self._encode(
                self._apply_template(messages[:i], add_generation_prompt=True)
            )
            full_until_asst_ids = self._encode(
                self._apply_template(messages[: i + 1], add_generation_prompt=False)
            )

            asst_start = len(prefix_ids)
            asst_end = len(full_until_asst_ids)
            asst_end = min(asst_end, len(input_ids))
            if asst_start < asst_end:
                labels[asst_start:asst_end] = input_ids[asst_start:asst_end]

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids_padded, attention_mask, labels_padded = [], [], []
    for x in batch:
        seq_len = len(x["input_ids"])
        pad_len = max_len - seq_len

        input_ids_padded.append(
            torch.cat(
                [
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                    x["input_ids"],
                ]
            )
        )
        attention_mask.append(
            torch.cat(
                [
                    torch.zeros(pad_len, dtype=torch.long),
                    torch.ones(seq_len, dtype=torch.long),
                ]
            )
        )
        labels_padded.append(
            torch.cat(
                [
                    torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                    x["labels"],
                ]
            )
        )

    return {
        "input_ids": torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels_padded),
    }


def save_model_and_tokenizer(accelerator, model, tokenizer, save_path: Path, is_main: bool):
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    unwrapped.save_pretrained(
        save_path,
        is_main_process=is_main,
        save_function=accelerator.save,
        state_dict=state_dict,
    )
    if is_main:
        tokenizer.save_pretrained(save_path)


def maybe_init_wandb(args, is_main: bool, task_name: str):
    if not args.use_wandb or not is_main:
        return None
    try:
        import wandb
    except ImportError:
        raise RuntimeError("wandb not installed. Run: pip install wandb")

    default_run = f"{task_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_name = args.wandb_run_name or default_run
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else []
    if task_name not in tags:
        tags.append(task_name)

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        tags=tags,
        config={
            "task": task_name,
            "model": args.model_name,
            "train_data": args.train_data,
            "val_data": args.val_data,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "max_seq_len": args.max_seq_len,
            "grad_checkpoint": args.grad_checkpoint,
            "flash_attn": args.flash_attn,
            "optimizer": "AdamW(betas=(0.9,0.95))",
            "scheduler": "cosine_with_warmup",
            "deepspeed": "ZeRO-2, no CPU offload",
        },
    )


def train(args, task_name: str) -> float:
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
    )

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    is_main = accelerator.is_main_process
    run = maybe_init_wandb(args, is_main, task_name=task_name)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print(f"Tokenizer loaded. pad_token = '{tokenizer.pad_token}'")
        print(f"Accelerator state: {accelerator.state}")

    use_flash_attn = args.flash_attn and importlib.util.find_spec("flash_attn") is not None
    if args.flash_attn and not use_flash_attn and is_main:
        print("Warning: flash_attn not installed, falling back to eager attention.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
    )

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if is_main:
            print("Gradient checkpointing enabled.")

    train_dataset = ReviewSFTDataset(args.train_data, tokenizer, args.max_seq_len)
    val_dataset = ReviewSFTDataset(args.val_data, tokenizer, args.max_seq_len)
    _collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    use_dist_sampler = accelerator.num_processes > 1
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
        )
        if use_dist_sampler
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=_collate,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if is_main:
        num_gpus = accelerator.num_processes
        eff_batch = args.batch_size * args.grad_accum * num_gpus
        print(f"\n{'=' * 55}")
        print(f"  Training config ({task_name})")
        print(f"{'=' * 55}")
        print(f"  Model:               {args.model_name}")
        print(f"  Num GPUs:            {num_gpus}")
        print(f"  Train samples:       {len(train_dataset)}")
        print(f"  Val samples:         {len(val_dataset)}")
        print(f"  Epochs:              {args.epochs}")
        print(f"  Batch size / GPU:    {args.batch_size}")
        print(f"  Grad accum steps:    {args.grad_accum}")
        print(f"  Effective batch:     {eff_batch}")
        print(f"  LR:                  {args.lr}")
        print(
            f"  Total optim steps:   {total_steps}"
            + (f"  (capped by --max_steps={args.max_steps})" if args.max_steps else "")
        )
        print(f"  Warmup steps:        {warmup_steps}")
        print(f"  Max seq len:         {args.max_seq_len}")
        print(f"  Flash attention:     {use_flash_attn}")
        print(f"{'=' * 55}\n")

        if run:
            run.config.update(
                {
                    "num_gpus": num_gpus,
                    "effective_batch": eff_batch,
                },
                allow_val_change=True,
            )

    best_val_loss = float("inf")
    global_step = 0
    hit_max_steps = False

    for epoch in range(1, args.epochs + 1):
        if hit_max_steps:
            break

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_tokens = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main)

        for batch in pbar:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            n_tokens = (batch["labels"] != IGNORE_INDEX).sum().item()
            train_loss += loss.detach().item() * n_tokens
            train_tokens += n_tokens

            if accelerator.sync_gradients:
                global_step += 1

                if args.max_steps and global_step >= args.max_steps:
                    hit_max_steps = True
                    if is_main:
                        print(f"\nReached max_steps={args.max_steps}, stopping training.")
                    break

                if is_main:
                    avg_loss = train_loss / max(train_tokens, 1)
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{current_lr:.2e}",
                            "step": global_step,
                        }
                    )

                    if run and global_step % max(1, args.wandb_log_steps) == 0:
                        run.log(
                            {
                                "train/loss": avg_loss,
                                "train/lr": current_lr,
                                "train/epoch": epoch,
                                "train/global_step": global_step,
                                "train/seen_tokens": train_tokens,
                            },
                            step=global_step,
                        )

                if args.save_steps and global_step % args.save_steps == 0:
                    ckpt_path = output_dir / f"checkpoint-step{global_step}"
                    save_model_and_tokenizer(accelerator, model, tokenizer, ckpt_path, is_main)
                    if is_main:
                        print(f"\nSaved checkpoint: {ckpt_path}")

        model.eval()
        val_loss = 0.0
        val_tokens = 0

        for batch in tqdm(val_loader, desc="Validation", disable=not is_main):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                n_tokens = (batch["labels"] != IGNORE_INDEX).sum().item()
                val_loss += outputs.loss.item() * n_tokens
                val_tokens += n_tokens

        def _reduce(value, dtype=torch.float32):
            tensor = torch.tensor([value], device=accelerator.device, dtype=dtype)
            return accelerator.reduce(tensor, reduction="sum").item()

        total_train_loss = _reduce(train_loss)
        total_train_tokens = _reduce(train_tokens, dtype=torch.long)
        total_val_loss = _reduce(val_loss)
        total_val_tokens = _reduce(val_tokens, dtype=torch.long)

        avg_train_loss = total_train_loss / max(total_train_tokens, 1)
        avg_val_loss = total_val_loss / max(total_val_tokens, 1)
        val_ppl = math.exp(min(avg_val_loss, 20))

        if is_main:
            print(
                f"\nEpoch {epoch} | "
                f"train_loss: {avg_train_loss:.4f} | "
                f"val_loss: {avg_val_loss:.4f} | "
                f"val_ppl: {val_ppl:.2f}"
            )
            if run:
                run.log(
                    {
                        "epoch/train_loss": avg_train_loss,
                        "epoch/val_loss": avg_val_loss,
                        "epoch/val_ppl": val_ppl,
                        "epoch/epoch": epoch,
                        "epoch/global_step": global_step,
                        "epoch/train_tokens": total_train_tokens,
                        "epoch/val_tokens": total_val_tokens,
                    },
                    step=global_step,
                )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = output_dir / "best_model"
            save_model_and_tokenizer(accelerator, model, tokenizer, best_path, is_main)
            if is_main:
                print(f"New best model saved: {best_path}  (val_loss={best_val_loss:.4f})")
                if run:
                    run.log(
                        {
                            "best/val_loss": best_val_loss,
                            "best/val_ppl": math.exp(min(best_val_loss, 20)),
                            "best/epoch": epoch,
                            "best/global_step": global_step,
                        },
                        step=global_step,
                    )

    final_path = output_dir / "final_model"
    save_model_and_tokenizer(accelerator, model, tokenizer, final_path, is_main)

    if is_main:
        print(f"\nFinal model saved: {final_path}")
        print(f"Best val loss: {best_val_loss:.4f}  (ppl: {math.exp(min(best_val_loss, 20)):.2f})")

    if is_main and run:
        run.summary["best_val_loss"] = best_val_loss
        run.summary["best_val_ppl"] = math.exp(min(best_val_loss, 20))
        run.finish()

    return best_val_loss


def build_parser(task_name: str, description: str, default_model_name: str = DEFAULT_MODEL_NAME):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--model_name", type=str, default=default_model_name)

    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps (0 = no limit)",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--grad_checkpoint", action="store_true", default=False)
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        default=False,
        help="Use Flash Attention 2 (requires flash-attn package)",
    )
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="ActReviewer")
    parser.add_argument("--wandb_entity", type=str, default="yiling210-yale-university")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, default=task_name)
    parser.add_argument("--wandb_log_steps", type=int, default=10)

    return parser


def run_cli(task_name: str, description: str, default_model_name: str = DEFAULT_MODEL_NAME):
    @record
    def _main():
        parser = build_parser(
            task_name=task_name,
            description=description,
            default_model_name=default_model_name,
        )
        args = parser.parse_args()
        train(args, task_name=task_name)

    _main()
