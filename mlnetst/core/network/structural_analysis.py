import torch
from mlnetst.utils.mlnet_metrics_utils import *
from pathlib import Path
import os

if __name__ == "__main__":
    print("DEBUGGING structural_analysis.py")
    sample_name = "mouse1_slice153"
    t = torch.load(Path(__file__).parents[3] / "data" / "processed" / f"{sample_name}_mlnet_sparse.pt")
    print(f"Loaded tensor shape: {t.shape}")
