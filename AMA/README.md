# AMA

**AMA** is the Agent-Driven Mention Augmenter in EVEMEL.

## Main Files

- `planner.py`: selects the appropriate augmentation tool and runs the executor.
- `tool.py`: executes the selected tool and writes tool-augmented outputs.
- `prompt/`: prompt templates for the planner and augmentation tools.
- `utils_glm.py`: MLLM API utility functions.

## Tool Names

The tool names are kept consistent with the paper and the generated data fields:

- `Default_Case`
- `Text_Supplement`
- `Symbolic_Solver`
- `OCR_Augment`
- `Visual_Grounder`

## Dependencies

AMA was tested with Python 3.10.20.

Install the AMA dependencies if you need to run the planner:

```bash
cd EVEMEL
pip install -r AMA/requirements.txt
```

Run both planner and executor:

```bash
cd EVEMEL/AMA
python planner.py --dataset WikiDiverse --mode test --stage all --data_root ../data/WikiDiverse --result_data_root ../result --api_key "$ZHIPUAI_API_KEY"
```
