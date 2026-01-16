
import torch
from data.reasoning_dataset import TransitiveReasoningDataset
from config import GPTConfig

def inspect_dataset():
    ds = TransitiveReasoningDataset(num_chains=10, chain_length=2, seed=42, is_test=False)
    
    # decode helper
    itos = {v:k for k,v in ds.node_to_id.items()}
    for k,v in ds.special_tokens.items():
        itos[v] = k
    
    print(f"Vocab size: {ds.vocab_size}")
    print(f"Sample data:")
    
    for i in range(3):
        x, y = ds[i]
        
        # Decode x
        x_tokens = [itos.get(t.item(), str(t.item())) for t in x if t.item() in itos]
        x_str = " ".join(x_tokens)
        
        # Decode y (only valid targets)
        y_valid_idx = (y != -100).nonzero(as_tuple=True)[0]
        y_targets = []
        for idx in y_valid_idx:
            val = y[idx].item()
            token = itos.get(val, str(val))
            y_targets.append(f"@{idx.item()}={token}")
            
        print(f"\nSample {i}:")
        print(f"Input: {x_str}")
        print(f"Targets: {', '.join(y_targets)}")
        
        # Check alignment
        # Input at target position should be '->'
        if len(y_valid_idx) > 0:
            last_valid = y_valid_idx[-1]
            input_at_pos = x[last_valid].item()
            input_char = itos.get(input_at_pos, '?')
            print(f"Input at target pos {last_valid}: {input_char}")

inspect_dataset()
