import torch
from mlnetst.utils.mlnet_metrics_utils import *
from pathlib import Path
import os

"""
Definition 1: Einstein product
- Let A\in\mathcal{R}^{I1\times\cdot\times IN\times J1\times\cdot\times JM}
- Let B\in\mathcal{R}^{J1\times\cdot\times JM\times K1\times\cdot\times KL}
be tensors of order N+M and M+L respectively.
The product would be C\in\mathcal{R}^{I1\times\cdot\times IN\times K1\times\cdot\times KL}

The elements of C will be a tensor of order N+L entries as:
\mathcal{C}_{i1,\ldots,iN,k1,\ldots,kL} = \sum_{j1,\ldots,jM} A_{i1,\ldots,iN,j1,\ldots,jM} B_{j1,\ldots,jM,k1,\ldots,kL}

This is commonly referred to as the Einstein product or tensor contraction.
"""


if __name__ == "__main__":
    print("DEBUGGING structural_analysis.py")
    sample_name = "mouse1_slice153"
    t = torch.load(Path(__file__).parents[3] / "data" / "processed" / f"{sample_name}_mlnet_sparse.pt")
    print(f"Loaded tensor shape: {t.shape}")
