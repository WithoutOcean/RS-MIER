# Thoughts Without Evidence Are Empty: A Visual-Evidence-First Principle for Multi-Image Remote Sensing Visual Question Answering

Chain-of-thought (CoT) provides an explicit textual trace of the visual evidence and intermediate reasoning steps involved in answering a question. However, existing methods typically optimize CoT as a unified sequence, without explicitly considering how to allocate the limited CoT token budget between visual evidence description and intermediate reasoning steps. In this work, we systematically investigate this allocation problem and find that, under a comparable CoT token budget, prioritizing sufficient and task-relevant visual evidence over extended reasoning leads to more accurate answers. We term this finding the Visual-Evidence-First Principle. Guided by this principle, we propose RS-MIER, an evidence-first CoT learning method for multi-image remote sensing visual question answering (VQA). Specifically, we construct MIVQA-30k, an evidence-driven multi-image CoT dataset, and employ evidence-dominant supervised fine-tuning (ED-SFT) to encourage explicit and comprehensive descriptions of visual evidence before reasoning. Furthermore, we propose Perception-Reasoning Decoupled Reinforcement Learning (PRD-RL), which introduces an independent reasoner to evaluate whether the generated visual evidence can independently support answering the question, thereby providing direct feedback on evidence quality. Combined with final-answer and correctness-state consistency rewards, PRD-RL jointly optimizes evidence construction and answer generation. Extensive experiments on our newly constructed benchmark, RS-MIST-Bench, and three public datasets, TEOChatlas, CDVQA, and ABCD, demonstrate the effectiveness of our method, achieving superior performance across these benchmarks.

## Overview

RS-MIER is implemented on top of two existing open-source training frameworks: [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) and [EasyR1](https://github.com/hiyouga/EasyR1). The `RS-MIER/LlamaFactory/` and `RS-MIER/easy-r1/` directories contain adapted copies of these upstream frameworks, respectively. The underlying framework implementations are credited to their original authors and contributors.

| Directory | Upstream framework | Role in RS-MIER |
| --- | --- | --- |
| `RS-MIER/LlamaFactory/` | [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) | Supervised fine-tuning, LoRA adapter training, and model export infrastructure for ED-SFT. |
| `RS-MIER/easy-r1/` | [EasyR1](https://github.com/hiyouga/EasyR1), which builds on verl | Reinforcement learning, rollout, and checkpoint infrastructure used by PRD-RL. |

Repository layout (relative to the repository root):

```text
.
├── README.md
├── RS-MIER/
│   ├── easy-r1/
│   └── LlamaFactory/
└── Demo_data/
    └── .gitkeep
```

`Demo_data/` is reserved for demonstration data. The `.gitkeep` file preserves this directory in Git until data files are added.

RS-MIER adapts these frameworks through task-specific training configurations, the evidence-first response format, and PRD-RL launch and reward integration with an independent reasoner. These project-specific configurations and integrations build on the upstream training infrastructure.

RS-MIER follows a two-stage training pipeline: evidence-dominant supervised fine-tuning (ED-SFT), followed by Perception-Reasoning Decoupled Reinforcement Learning (PRD-RL).

- `RS-MIER/LlamaFactory/` provides the supervised fine-tuning and model export framework.
- `RS-MIER/LlamaFactory/examples/train_lora/qwen3vl_lora_sft.yaml` configures Qwen3-VL LoRA fine-tuning.
- `RS-MIER/LlamaFactory/examples/merge_lora/qwen3vl_lora_sft.yaml` merges the trained LoRA adapter into a complete model.
- `RS-MIER/easy-r1/examples/qwen3_vl_RS_MIER.sh` launches PRD-RL training and manages the independent reasoner services.
- `RS-MIER/easy-r1/examples/reward_function/reward_four_part.py` implements the format, visual-evidence, correctness-state consistency, and final-answer rewards.
- `RS-MIER/easy-r1/examples/format_prompt/multi_image.jinja` defines the evidence-first response format.
- `RS-MIER/easy-r1/scripts/model_merger.py` exports distributed RL checkpoints.

The model generates its response in the following order:

```text
<perception>Task-relevant visual evidence from the input images.</perception>
<think>Reasoning based on the visual evidence.</think>
<answer>Final answer.</answer>
```

During PRD-RL, an independent reasoner receives the question and the generated visual evidence. Its ability to answer the question provides feedback on evidence quality.

## Installation

The commands below assume Linux, CUDA, NVIDIA GPUs, and a checkout at `/path/to/RS-MIER`. Use Python 3.11 for the environments below; the bundled LlamaFactory requires Python 3.11 or later. Install a PyTorch build compatible with your CUDA environment.

```bash
git clone https://github.com/WithoutOcean/RS-MIER.git
cd RS-MIER
```

Use separate Python environments for ED-SFT and PRD-RL because the two frameworks declare different dependency ranges.

### Environment 1: ED-SFT

```bash
conda create -n rs-mier-sft python=3.11 -y
conda activate rs-mier-sft
cd /path/to/RS-MIER/RS-MIER/LlamaFactory
python -m pip install -e .
```

### Environment 2: PRD-RL

```bash
conda create -n rs-mier-rl python=3.11 -y
conda activate rs-mier-rl
cd /path/to/RS-MIER/RS-MIER/easy-r1
# Install the CUDA-compatible PyTorch build for your system first.
python -m pip install -e .
```

PRD-RL uses vLLM, Ray, FlashAttention, and Qwen-VL utilities. The declared dependencies are in `RS-MIER/easy-r1/requirements.txt`. Use versions that support your Qwen3-VL checkpoint and installed CUDA/PyTorch stack.

## Training

### Data

The data used in RS-MIER cover supervised fine-tuning, reinforcement learning, and evaluation:

1. **Evidence-driven CoT data (MIVQA-30k):** Multi-image remote sensing questions with visual evidence descriptions, reasoning, and final answers. This dataset supports ED-SFT to enrich the model's visual perception and encourage more comprehensive descriptions of task-relevant evidence.
2. **Reinforcement learning data:** Multi-image remote sensing questions with reference final answers, used for policy optimization and reward computation during PRD-RL.
3. **Evaluation data:** Our constructed RS-MIST-Bench and the public TEOChatlas, CDVQA, and ABCD datasets, used to evaluate multi-image remote sensing question answering.

**Data availability.** MIVQA-30k and RS-MIST-Bench will be made publicly available in a future release.

### Models

Prepare the base vision-language model and the independent reasoner locally.

```text
checkpoints/
├── Qwen3-VL-4B-Instruct/   # base model for ED-SFT
├── ED-SFT/                # merged ED-SFT model used to initialize PRD-RL
├── Qwen3-32B/             # independent text reasoner
└── RS-MIER/               # exported model after PRD-RL
```

Each model directory should contain the complete checkpoint and its tokenizer/configuration files. Vision-language checkpoints also need the corresponding processor files.

### Stage 1: Evidence-dominant supervised fine-tuning

ED-SFT uses the bundled LLaMA-Factory framework to fine-tune a pretrained vision-language model for evidence-first multi-image remote sensing question answering. The resulting checkpoint initializes the policy model for the subsequent PRD-RL stage.

#### Dataset

ED-SFT uses MIVQA-30k, an evidence-driven multi-image CoT dataset, to enrich the model's visual perception. The supervision emphasizes detailed and comprehensive descriptions of task-relevant visual information, encouraging the model to capture and articulate richer visual evidence before reasoning.

- **Images and questions:** Multiple remote sensing images paired with a question about their visual content.
- **Target responses:** An evidence-first response organized as `<perception>...</perception><think>...</think><answer>...</answer>`, covering visual evidence, reasoning, and the final answer in order.

### Stage 2: Perception-Reasoning Decoupled Reinforcement Learning

PRD-RL initializes the policy model from the ED-SFT checkpoint and optimizes it with Group Relative Policy Optimization (GRPO) using EasyR1. For each multi-image question, the policy generates multiple responses in the evidence–reasoning–answer format and is updated according to their relative rewards.

The combined reward accounts for response format, visual-evidence quality, correctness-state consistency, and final-answer accuracy. This stage encourages the policy model to construct task-relevant visual evidence that supports answering the question and to produce more accurate final answers.

#### Judge model

During PRD-RL training, Qwen3-32B is additionally loaded as an independent judge model. Given the question and the visual evidence generated by the policy model, it produces an answer that is used to evaluate evidence quality and compute the corresponding training rewards.

#### Dataset

Prepare local training and validation files with `problem`, `answer`, and `images` fields. The loader supports JSON, JSONL, and Parquet. A minimal JSONL record is:

```json
{"problem":"<image><image>\nDid the building area increase between the two observations?","answer":"yes","images":["sample/before.png","sample/after.png"]}
```

- `problem` contains the question and image placeholders.
- `answer` contains the reference final answer.
- `images` contains image paths in the order expected by the question. Relative paths are resolved against `IMAGE_DIR`.

Preserve the acquisition order when preparing temporal image pairs. The loader follows the supplied list and does not reorder images by timestamp. The prompt template automatically appends the instructions for generating `<perception>`, `<think>`, and `<answer>`.

#### Configuration and start

Before starting training, configure `MODEL_PATH`, `JUDGE_MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`, and `IMAGE_DIR` in `RS-MIER/easy-r1/examples/qwen3_vl_RS_MIER.sh`, or supply them as environment variables as shown below.

From the repository root (the directory containing this README), activate the PRD-RL environment, enter `RS-MIER/easy-r1/`, and launch training:

```bash
conda activate rs-mier-rl
cd RS-MIER/easy-r1
bash examples/qwen3_vl_RS_MIER.sh
```

For example, if model checkpoints are stored under `RS-MIER/easy-r1/checkpoints/` and data under `RS-MIER/easy-r1/dataset/`, run the following command from `RS-MIER/easy-r1/`. Adjust these relative paths to match your directory layout:

```bash
MODEL_PATH="./checkpoints/ED-SFT" \
JUDGE_MODEL_PATH="./checkpoints/Qwen3-32B" \
TRAIN_FILE="./dataset/train.jsonl" \
VAL_FILE="./dataset/val.jsonl" \
IMAGE_DIR="./dataset/images" \
bash examples/qwen3_vl_RS_MIER.sh
```

The launcher starts the reasoner services, waits for them to become ready, launches training, and shuts down the services when training exits. The main settings are:

```text
Algorithm:      GRPO
Training:       epochs=2, actor_learning_rate=1e-6, KL_coefficient=1e-2
Token limits:   max_prompt_length=4096, max_response_length=512
Rollout:        ROLLOUT_BATCH_SIZE=32, MINI_ROLLOUT_BATCH_SIZE=4, ROLLOUT_N=8
Actor batches:  ACTOR_GLOBAL_BATCH_SIZE=8, ACTOR_MICRO_BATCH_SIZE=1
Images:         up to 5 images per rollout prompt
Validation:     VAL_BATCH_SIZE=16, VAL_FREQ=300
Checkpointing:  every 600 steps
```

The uppercase batch and validation variables can be set before the launch command. Additional settings are defined in `RS-MIER/easy-r1/examples/config.yaml` and overridden by the project launcher where applicable.

#### Export the RL checkpoint

With the default experiment name, actor checkpoints are saved under:

```text
RS-MIER/easy-r1/checkpoints/easy_r1/multi_image_reason/global_step_<step>/actor/
```

Replace `<step>` with the checkpoint step to export:

```bash
cd /path/to/RS-MIER/RS-MIER/easy-r1

python scripts/model_merger.py \
  --local_dir "/path/to/RS-MIER/RS-MIER/easy-r1/checkpoints/easy_r1/multi_image_reason/global_step_<step>/actor"
```

The exporter writes the Hugging Face model to the actor directory's `huggingface/` subdirectory.

## Acknowledgements

We thank the authors and contributors of [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) and [EasyR1](https://github.com/hiyouga/EasyR1) for their open-source training frameworks. RS-MIER uses and adapts these frameworks for its supervised fine-tuning and reinforcement learning stages.

The bundled framework code remains subject to its upstream copyright notices and license terms. See [RS-MIER/LlamaFactory/LICENSE](https://github.com/WithoutOcean/RS-MIER/blob/main/RS-MIER/LlamaFactory/LICENSE) and [RS-MIER/easy-r1/LICENSE](https://github.com/WithoutOcean/RS-MIER/blob/main/RS-MIER/easy-r1/LICENSE) for the respective Apache License 2.0 texts. Model checkpoints and datasets are subject to their own licenses and usage terms.
