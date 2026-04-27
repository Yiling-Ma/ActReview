# ACTREVIEW: A Review-to-Revision Framework for Actionable Peer Review Generation

<p align="center">
  <b>Paper 📄</b> · <a href="#">ActReview</a> |
  <b>Dataset 🐙</b> · <a href="https://huggingface.co/datasets/YilingMa/ActReview-40K">ActReview-40K</a> |
  <b>Code 💻</b> · <a href="https://github.com/Yiling-Ma/ActReview">ActReview</a>
</p>

## Overview
ActReview studies **actionable peer-review generation**: instead of only identifying weaknesses in a paper, the model should also produce revision-oriented guidance that authors can directly act on. Following the paper, the repository formulates the problem as two linked subtasks:

- **Task 1: Claim Generation**
  Given a paper and a target weakness category, generate one or more precise diagnostic claims describing what is wrong.
- **Task 2: Actionable Suggestion Generation**
  Given a claim and paper context, generate grounded suggestions specifying **what** to revise, **where** to revise it, **how** to do it, and the expected outcome.


## Repository Layout

```text
ActReview/
├── Data_Collection/
│   ├── get_iclr_raw_data.py
│   ├── get_icml_raw_data.py
│   ├── get_emnlp_raw_data.py
│   ├── align_weakness_rebuttal.py
│   ├── classify_weakness.py
│   ├── generate_enhanced_reviews.py
│   ├── align_snippets_dual_task.py
│   └── group_by_paper_l2.py
├── SFT/
│   ├── convert_to_sft.py
│   ├── sft_train_task1.py
│   ├── sft_train_task2.py
│   ├── sft_train_common.py
│   └── inference.py
└── Rubric_RL/
    ├── stage1_sft_candidates.py
    ├── stage2_gpt_candidates.py
    ├── stage3_extract_rubrics.py
    ├── prepare_verl_data.py
    ├── rubric_reward_verl_task1.py
    ├── rubric_reward_verl_task2.py
    ├── inference_task1_grpo.py
    └── inference_task2_grpo.py
```


## Data Construction Pipeline

### 1. Collect raw multi-turn review threads

The repository includes separate collectors for ICLR, ICML, and EMNLP OpenReview data:

- [get_iclr_raw_data.py](Data_Collection/get_iclr_raw_data.py)
- [get_icml_raw_data.py](Data_Collection/get_icml_raw_data.py)
- [get_emnlp_raw_data.py](Data_Collection/get_emnlp_raw_data.py)

Each script outputs one JSONL row per submission with flattened review-thread events.

Typical usage:

```bash
python Data_Collection/get_iclr_raw_data.py --help
python Data_Collection/get_icml_raw_data.py --help
python Data_Collection/get_emnlp_raw_data.py --help
```

### 2. Align review weaknesses with rebuttal segments

[align_weakness_rebuttal.py](Data_Collection/align_weakness_rebuttal.py) decomposes review text into atomic weakness points and maps each weakness to the rebuttal span that addresses it.

This script uses **Azure OpenAI** and expects environment variables such as:

```bash
export AZURE_OPENAI_ENDPOINT=...
export AZURE_OPENAI_KEY=...
export AZURE_OPENAI_DEPLOYMENT=...
export AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

### 3. Classify weaknesses into the taxonomy

[classify_weakness.py](Data_Collection/classify_weakness.py) assigns each weakness to the paper's taxonomy of **7 L1 categories** and **17 L2 categories**. The classifier writes a `weakness_category` field with label ids, names, confidence, and reasoning.

### 4. Generate enhanced actionable reviews

[generate_enhanced_reviews.py](Data_Collection/generate_enhanced_reviews.py) rewrites vague weaknesses into structured actionable peer-review objects:

- `claim`
- `evidence`
- `actionable_suggestions`
- `citations`
- `severity`

The script includes validation to prevent:

- rebuttal leakage,
- unsupported section references when section indices are missing,
- vague `where` fields,
- schema drift outside the target template.

### 5. Retrieve task-specific paper snippets

[align_snippets_dual_task.py](Data_Collection/align_snippets_dual_task.py) downloads PDFs, extracts text, and attaches localized evidence for both tasks:

- `aligned_snippets_task1`
- `aligned_snippets_task2_evidence`
- `aligned_snippets_task2_support`
- `aligned_snippets_task2`

Typical usage:

```bash
python Data_Collection/align_snippets_dual_task.py \
  --input enhanced_reviews.jsonl \
  --output enhanced_data_with_snippets_dual.jsonl \
  --pdf_cache_dir data/pdf_cache \
  --top_k_task1 3 \
  --top_k_task2_final 5 \
  --top_k_task2_channel 4
```


## Data Format

The normalized dataset follows a single per-instance schema with:

- paper metadata,
- original weakness,
- follow-ups and rebuttals,
- weakness taxonomy label,
- enhanced review,
- metadata about the discussion,
- task-specific aligned snippets.

The main fields are:

```json
{
  "weakness_id": "...",
  "paper_context": {
    "title": "...",
    "abstract": "...",
    "keywords": ["..."],
    "pdf_url": "...",
    "web_url": "...",
    "sections_index": [{"section_id": "s1", "name": "Introduction", "page": 1}]
  },
  "original_weakness": "...",
  "follow_ups": ["..."],
  "rebuttals": ["..."],
  "weakness_category": {
    "l1_id": "L1.2",
    "l1_name": "...",
    "l2_id": "L2.2.1",
    "l2_name": "...",
    "confidence": 0.94,
    "reasoning": "..."
  },
  "enhanced_review": {
    "claim": "...",
    "evidence": "...",
    "actionable_suggestions": [
      {
        "what": "...",
        "where": "...",
        "how": "...",
        "expected_outcome": "...",
        "priority": "critical"
      }
    ],
    "citations": [],
    "severity": "major"
  }
}
```

## Supervised Fine-Tuning

### Convert the dataset into chat-format SFT data

[convert_to_sft.py](SFT/convert_to_sft.py) transforms classified enhanced reviews into the multi-turn training format expected by Qwen-style chat fine-tuning.

It creates records like:

1. system prompt
2. user prompt for Task 1
3. assistant claim
4. user prompt for Task 2
5. assistant evidence + suggestions + severity

Typical usage:

```bash
python SFT/convert_to_sft.py \
  --input classified_enhanced_reviews.jsonl \
  --output_dir sft_data
```

### Train Task 1 and Task 2 models

The repository trains **dedicated models** for the two tasks:

- [sft_train_task1.py](SFT/sft_train_task1.py)
- [sft_train_task2.py](SFT/sft_train_task2.py)

Both use [sft_train_common.py](SFT/sft_train_common.py), which supports:

- Qwen/Qwen3-8B-Base initialization,
- bf16 training,
- gradient accumulation,
- gradient checkpointing,
- optional FlashAttention 2,
- optional Weights & Biases logging.

Example commands:

```bash
python SFT/sft_train_task1.py \
  --train_data sft_data/task1_train.jsonl \
  --val_data sft_data/task1_val.jsonl \
  --output_dir checkpoints/task1 \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 4 \
  --lr 1e-5 \
  --grad_checkpoint \
  --flash_attn
```

```bash
python SFT/sft_train_task2.py \
  --train_data sft_data/task2_train.jsonl \
  --val_data sft_data/task2_val.jsonl \
  --output_dir checkpoints/task2 \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 4 \
  --lr 1e-5 \
  --grad_checkpoint \
  --flash_attn
```

## SFT Inference

[inference.py](SFT/inference.py) runs inference for:

- `task1`
- `task2`
- `both`

In the current implementation, Task 2 evaluation uses **ground-truth claims as oracle inputs**.

Example:

```bash
python SFT/inference.py \
  --model_path checkpoints/task2/best_model \
  --input_path test.jsonl \
  --output_file outputs/sft_predictions.jsonl \
  --task both
```

## Rubric-Guided RL

The RL pipeline is organized as an offline rubric-construction stage followed by verl-compatible GRPO training.

### Stage 1: Generate SFT candidates

[stage1_sft_candidates.py](Rubric_RL/stage1_sft_candidates.py) runs the SFT model over rubric specs and caches candidate outputs. It supports multi-GPU sharding through `--num_workers` and `--gpu_ids`.

```bash
python Rubric_RL/stage1_sft_candidates.py \
  --specs rubric_specs.jsonl \
  --sft_checkpoint checkpoints/task2/best_model \
  --output artifacts/sft_candidates.jsonl \
  --artifact_dir artifacts
```

### Stage 2: Generate GPT candidates

[stage2_gpt_candidates.py](Rubric_RL/stage2_gpt_candidates.py) generates stronger or more diverse reference candidates using OpenAI models, either through synchronous API calls or the Batch API.

Expected environment:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.1
```

Example:

```bash
python Rubric_RL/stage2_gpt_candidates.py \
  --specs rubric_specs.jsonl \
  --sft_candidates artifacts/sft_candidates.jsonl \
  --output artifacts/gpt_candidates.jsonl \
  --artifact_dir artifacts
```

### Stage 3: Extract weakness-specific rubrics

[stage3_extract_rubrics.py](Rubric_RL/stage3_extract_rubrics.py) builds instance-level rubrics from the spec, SFT candidates, and GPT candidates. It can also generate verifier functions for format-oriented requirements.

```bash
python Rubric_RL/stage3_extract_rubrics.py \
  --specs rubric_specs.jsonl \
  --sft_candidates artifacts/sft_candidates.jsonl \
  --gpt_candidates artifacts/gpt_candidates.jsonl \
  --output artifacts/final_rubrics.jsonl \
  --artifact_dir artifacts
```

### Convert rubrics into verl data

[prepare_verl_data.py](Rubric_RL/prepare_verl_data.py) converts the final rubric JSONL into parquet files used by verl GRPO training.

```bash
python Rubric_RL/prepare_verl_data.py \
  --rubrics artifacts/final_rubrics.jsonl \
  --output_dir verl_data \
  --train_ratio 0.95
```

### Reward functions

The task-specific reward implementations are:

- [rubric_reward_verl_task1.py](Rubric_RL/rubric_reward_verl_task1.py)
- [rubric_reward_verl_task2.py](Rubric_RL/rubric_reward_verl_task2.py)

These files implement rubric-based soft scoring with additional structure, repetition, and formatting penalties. They also maintain persistent SQLite caches for judge outputs to reduce repeated API cost.

## GRPO Inference

After RL training, the repository provides separate inference scripts:

- [inference_task1_grpo.py](Rubric_RL/inference_task1_grpo.py)
- [inference_task2_grpo.py](Rubric_RL/inference_task2_grpo.py)

Task 1 inference generates weakness claims from the paper context and weakness label. Task 2 inference uses ground-truth claims as input and evaluates the RL-tuned suggestion generator.

## Environment and Dependencies

The repository now includes a dependency file: [requirements.txt](requirements.txt).

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies included in `requirements.txt`:

- Python 3.10+
- `torch`
- `transformers`
- `accelerate`
- `deepspeed`
- `pandas`
- `pyarrow`
- `tqdm`
- `requests`
- `openai`
- `PyMuPDF`
- `openreview-py`
- `wandb`

Notes:

- `flash-attn` is not pinned in `requirements.txt` because installation depends on the local CUDA and PyTorch stack.
- `verl` is also not pinned here; this repository includes verl data preparation and reward functions, but GRPO training environments are often managed separately.

## Notes on Reproducibility

Several scripts rely on external services and credentials:

- **Azure OpenAI** for review segmentation, weakness-rebuttal alignment, and some data-generation stages.
- **OpenAI API** for GPT candidate generation and rubric-based reward scoring.
- **OpenReview PDFs** for snippet retrieval and evidence grounding.

Because of that, reproducing the full pipeline from raw review threads requires:

- valid API credentials,
- network access,
- access to the same or equivalent OpenReview data snapshots.

## Citation

If you use this repository or the ActReview dataset, please cite the ActReview paper.

```bibtex
@article{ma2026actreview,
  title={ACTREVIEW: A Review-to-Revision Framework for Actionable Peer Review Generation},
  author={Ma, Yiling and Zhao, Yilun and Wu, Sihong and Chen, Ziyu and Patwardhan, Manasi and Cohan, Arman},
  journal={arXiv},
  year={2026}
}
```
