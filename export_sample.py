
import os
import torch
from data.reasoning_dataset import TransitiveReasoningDataset

def export_sample(filename="dataset_sample.txt", num_samples=20):
    ds = TransitiveReasoningDataset(num_chains=num_samples, chain_length=2, seed=42, is_test=False)
    
    itos = {v:k for k,v in ds.node_to_id.items()}
    for k,v in ds.special_tokens.items():
        itos[v] = k
        
    print(f"Exporting {num_samples} samples to {filename}...")
    
    with open(filename, "w") as f:
        f.write(f"Dataset Sample (Max Len: {len(ds[0][0])})\n")
        f.write("="*50 + "\n\n")
        
        for i in range(num_samples):
            x, y = ds[i]
            
            # Decode x
            x_tokens = [itos.get(t.item(), str(t.item())) for t in x]
            x_str = " ".join(x_tokens)
            
            # Decode y (target) - we only care about the final prediction usually
            # But let's show the non-ignored ones
            y_valid = (y != -100)
            if y_valid.any():
                last_idx = y_valid.nonzero()[-1].item()
                target_val = y[last_idx].item()
                target_token = itos.get(target_val, str(target_val))
            else:
                target_token = "None"
                
            f.write(f"Sample {i+1}:\n")
            f.write(f"Input:  {x_str}\n")
            f.write(f"Target: {target_token} (at pos {last_idx if y_valid.any() else -1})\n")
            f.write("-" * 20 + "\n")
            
    print("Done.")

if __name__ == "__main__":
    export_sample()
