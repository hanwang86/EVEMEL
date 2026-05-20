# HVMCM

**HVMCM** is the Hierarchical Visual MoE based Cross-modal Matching module used by the EVEMEL model.

## Main Files

- `main.py`: training and testing entry.
- `model/modeling_eve.py`: HVMCM, expert routing, token-level MoE fusion, and Bidirectional Gated Intra-modal Matching (BiGIM).
- `model/lightning_eve.py`: Lightning module for the joint training strategy.
- `utils/dataset_ama.py`: dataset module that consumes AMA tool outputs.
- `utils/dataset.py`: dataset module for non-tool-augmented inputs.
- `utils/loss.py`: hard-negative contrastive loss and BiGIM auxiliary contrastive loss.
- `hard_negatives.py`: hard negative mining script.

## Dependencies

HVMCM was tested with Python 3.7.16.

Install this module's environment with:

```bash
pip install -r requirements.txt
```

If `torch==1.11.0+cu113` is not resolved by the default PyPI index, install PyTorch from the CUDA 11.3 wheel index first, then install the remaining packages.
