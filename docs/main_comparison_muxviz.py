import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import subprocess
import pandas as pd

from mlnetst.utils.mlnet_utils import build_edgelist_from_tensor
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Compare with muxviz.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="smallest",
        help="Name of the experiment for saving results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for edge list file. Defaults to data/tmp/",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the tensor
    tensor_path = Path(__file__).resolve().parents[1] / "data" / "processed" / f"{args.experiment_name}_mlnet.pth"
    print(f"Loading tensor from: {tensor_path}")
    
    if not tensor_path.exists():
        print(f"Error: Tensor file {tensor_path} does not exist.")
        return
    
    t = torch.load(str(tensor_path))
    print(f"Loaded tensor with shape: {t.shape}")
    
    # Build edge list
    print("Building edge list...")
    edge_list = build_edgelist_from_tensor(t)

    # Print edge list summary
    print(f"Generated edge list with {len(edge_list)} edges")
    
    # Convert to pandas DataFrame if it isn't already
    if not isinstance(edge_list, pd.DataFrame):
        # Assuming edge_list is a list of tuples or similar structure
        # Adjust column names based on your actual edge list format
        if hasattr(edge_list, '__len__') and len(edge_list) > 0:
            if len(edge_list[0]) == 4:  # Assuming (node.from, layer.from, node.to, layer.to, weight)
                edge_df = pd.DataFrame(edge_list, columns=['node.from', 'layer.from', 'node.to', 'layer.to', 'weight'])
            elif len(edge_list[0]) == 3:  # Assuming (node.from, layer.from, node.to, layer.to)
                edge_df = pd.DataFrame(edge_list, columns=['node.from', 'layer.from', 'node.to', 'layer.to'])
            else:
                edge_df = pd.DataFrame(edge_list)
        else:
            print("Error: Empty or invalid edge list")
            return
    else:
        edge_df = edge_list

    # Add 1 to node and layer indices to match muxviz format
    edge_df = edge_df.transform(lambda x: x + 1 if x.dtype in [int, float] else x)
    
    print("Edge list DataFrame:")
    print(edge_df.head())
    print(f"Shape: {edge_df.shape}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parents[1] / "data" / "tmp"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save edge list as CSV
    edgelist_file = output_dir / f"{args.experiment_name}_edgelist.csv"
    print(f"Saving edge list to: {edgelist_file}")
    edge_df.to_csv(edgelist_file, index=False)
    
    # Call R script
    r_script_path = Path(__file__).parent / "main_comparison_muxviz.R"
    
    if not r_script_path.exists():
        print(f"Error: R script {r_script_path} does not exist.")
        return
    
    print(f"Calling R script: {r_script_path}")
    print(f"With argument: {edgelist_file}")
    
    try:
        # Run R script with subprocess
        result = subprocess.run([
            "Rscript", 
            str(r_script_path), 
            str(edgelist_file)
        ], 
        capture_output=True, 
        text=True, 
        timeout=300  # 5 minute timeout
        )
        
        print("\n" + "="*50)
        print("R SCRIPT OUTPUT:")
        print("="*50)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✓ R script executed successfully!")
        else:
            print("✗ R script failed!")
            
    except subprocess.TimeoutExpired:
        print("Error: R script timed out after 5 minutes")
    except FileNotFoundError:
        print("Error: Rscript not found. Make sure R is installed and in your PATH.")
    except Exception as e:
        print(f"Error running R script: {e}")

if __name__ == "__main__":
    main()