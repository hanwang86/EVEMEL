# Learning to Better Exploit Visual Evidence for Multimodal Entity Linking
## Overview
This repository contains the code for **Learning to Better Exploit Visual Evidence for Multimodal Entity Linking**. The proposed model, **EVEMEL**, learns to better exploit visual evidence through an adaptive mention augmentation stage (**AMA**) and a hierarchical visual MoE based cross-modal matching stage (**HVMCM**) with **Bidirectional Gated Intra-modal Matching (BiGIM)** and hard negative training.
<p align="center">
  <img src="figures/overview.pdf" alt="Overview of EVEMEL" width="900"/>
</p>
## Repository Structure

```text
EVEMEL/
|-- AMA/                         # Agent-Driven Mention Augmenter
|   |-- prompt/                  # planner and tool prompts
|   |-- planner.py               # AMA planner and executor entry
|   |-- tool.py                  # AMA tool executor
|   |-- utils_glm.py             # MLLM API utilities
|   |-- README.md
|   `-- requirements.txt         # AMA-only extra dependencies
|-- HVMCM/                       # Hierarchical Visual MoE based Cross-modal Matching
|   |-- config/                  # dataset and training YAML configs
|   |-- model/
|   |   |-- lightning_eve.py     # Lightning training module
|   |   `-- modeling_eve.py      # HVMCM, visual expert routing, MoE fusion, and BiGIM
|   |-- utils/
|   |   |-- dataset_ama.py       # AMA-augmented mention input construction
|   |   |-- dataset.py           # raw mention/entity input construction
|   |   `-- loss.py              # hard-negative and BiGIM contrastive losses
|   |-- hard_negatives.py        # hard negative mining
|   |-- main.py                  # train/test entry
|   |-- README.md
|   `-- requirements.txt         # HVMCM environment
`-- .gitignore
```

## Environment

EVEMEL uses separate environments for AMA and HVMCM:

- AMA: Python 3.10.20
- HVMCM: Python 3.7.16

For the main HVMCM training environment:

```bash
pip install -r HVMCM/requirements.txt
```

Install the AMA dependencies only when running the augmentation stage:

```bash
pip install -r AMA/requirements.txt
```

## Data and Checkpoints

The raw datasets used in this project follow the datasets released with the [MIMIC](https://github.com/pengfei-luo/MIMIC) paper. Prepare the data locally before running either AMA or HVMCM:

1. Create `./data` under the repository root and place the MIMIC-format datasets there, for example `./data/WikiMEL`, `./data/RichpediaMEL`, and `./data/WikiDiverse`.
2. If a config uses a description-enhanced entity file such as `kb_entity_desc.json`, place that file in the corresponding dataset folder and update the YAML path accordingly.
3. Download [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).
4. Create `./checkpoint` under the repository root and place the pretrained CLIP weight there, or update the YAML config field `pretrained_model` to the Hugging Face model id or your local checkpoint path.

Large datasets, pretrained weights, logs, caches, and checkpoints are intentionally not included in this repository.

## AMA Requirements

AMA uses two external visual tools mentioned in the paper:

- [GroundingDINO](https://huggingface.co/IDEA-Research/grounding-dino-base) for the `Visual_Grounder` tool.
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for the `OCR_Augment` tool.

`AMA/tool.py` loads GroundingDINO from `GROUNDING_DINO_MODEL_PATH` when the environment variable is set; otherwise it uses `IDEA-Research/grounding-dino-base`. PaddleOCR is installed through `AMA/requirements.txt`.

## Usage

Run AMA first if you want to train or evaluate HVMCM with AMA-augmented inputs. The `*_ama.yaml` configs in `HVMCM/config` expect AMA outputs such as `*_with_tools.json` and `grounded_images`, so those files must be generated before using the AMA version of the model.

Run AMA:

```bash
cd EVEMEL/AMA
python planner.py --dataset WikiDiverse --mode test --stage all --data_root ../data/WikiDiverse --result_data_root ../result --api_key "$ZHIPUAI_API_KEY"
```

Run HVMCM with AMA outputs:

```bash
cd EVEMEL/HVMCM
python main.py --config ./config/wikidiverse_ama.yaml
```

Run HVMCM without AMA augmentation:

```bash
cd EVEMEL/HVMCM
python main.py --config ./config/wikidiverse.yaml
```

Mine hard negatives:

```bash
cd EVEMEL/HVMCM
python hard_negatives.py --entity_file /path/to/kb_entity.json --train_file /path/to/train.json --output_path /path/to/hard_negatives.json
```

## Notes

- Set `ZHIPUAI_API_KEY` before running AMA, or pass `--api_key` explicitly.
- Update all dataset, AMA output, hard-negative, checkpoint, and device paths in the YAML configs before training or evaluation.
- For AMA-based experiments, keep the paths in `HVMCM/config/*_ama.yaml` aligned with the `--result_data_root` used by AMA.
