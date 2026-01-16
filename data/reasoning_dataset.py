import torch
import random
import numpy as np

class TransitiveReasoningDataset:
    def __init__(self, num_chains=400, chain_length=2, seed=42, is_test=False):
        """
        Args:
            num_chains: Number of distinct chains to generate
            chain_length: Number of hops (2 or 3)
            seed: Random seed for reproducibility
            is_test: If True, generates held-out test set
        """
        self.nodes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # 26 nodes
        self.node_to_id = {node: i for i, node in enumerate(self.nodes)}
        self.special_tokens = {'→': 26, '.': 27, 'Q': 28, '?': 29, '[PAD]': 30}
        self.vocab_size = 32  # 26 nodes + 5 special + 1 reserved? Max is 32.
        # Let's map exactly as specified: A-Z (0-25), special (26-30).
        # We need 32 tokens total? Spec says "vocabulary = 32 tokens".
        # 26 + 5 = 31. We can leave 31 unused or for [EOS] if needed.
        
        self.chain_length = chain_length
        self.seed = seed
        self.data = []
        
        self._generate_dataset(num_chains, is_test)

    def _generate_dataset(self, num_chains, is_test):
        # We need to split on *complete chains*.
        # Total set of possible chains is huge, but we want controlled splitting.
        # Strategy: Generate a large pool of valid chains deterministically sorted, then split.
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        all_chains = []
        
        if self.chain_length == 2:
            # A->B, B->C
            for a in self.nodes:
                for b in self.nodes:
                    if a == b: continue
                    for c in self.nodes:
                        if b == c or a == c: continue
                        all_chains.append([a, b, c])
        elif self.chain_length == 3:
            # A->B, B->C, C->D
            for a in self.nodes:
                for b in self.nodes:
                    if a == b: continue
                    for c in self.nodes:
                        if b == c or a == c: continue
                        for d in self.nodes:
                             if c == d or b == d or a == d: continue
                             all_chains.append([a, b, c, d])
        
        # Shuffle deterministically
        random.shuffle(all_chains)
        
        # Split 80/20 for 2-hop
        split_idx = int(0.8 * len(all_chains))
        
        if self.chain_length == 2:
            start_chains = all_chains[:split_idx] if not is_test else all_chains[split_idx:]
            # Subsample to requested num_chains
            if len(start_chains) > num_chains:
                start_chains = start_chains[:num_chains]
                
            self.chains = start_chains
        else:
            # For 3-hop (OOD), we just take the first N (spec says "Separately generate... not used in training")
            self.chains = all_chains[:num_chains]
            
    def __len__(self):
        return len(self.chains)
    
    def __getitem__(self, idx):
        chain = self.chains[idx]
        return self._tokenize_chain(chain)
        
    def _tokenize_chain(self, chain):
        # Format: [A][→][B][.][B][→][C][.][Q][A][?][→]... Target: [C]
        # For 3-hop: [A][→][B][.][B][→][C][.][C][→][D][.][Q][A][?][→]... Target: [D]
        
        seq = []
        
        # Input Chain context
        for i in range(len(chain) - 1):
            src = chain[i]
            tgt = chain[i+1]
            seq.extend([self.node_to_id[src], self.special_tokens['→'], self.node_to_id[tgt], self.special_tokens['.']])
            
        # Query
        start_node = chain[0]
        end_node = chain[-1]
        
        seq.extend([self.special_tokens['Q'], self.node_to_id[start_node], self.special_tokens['?'], self.special_tokens['→']])
        
        # Target (for training, we append it to input for autoregressive loss)
        # But wait, standard GPT training inputs are X, targets are shifted.
        # Spec:
        # Input:  [A][→][B][.][B][→][C][.][Q][A][?][→]
        # Target: [C]
        # For a standard GPT, we usually feed "Input + Target" and mask loss.
        # So full sequence is ...[→][C]
        
        full_seq = seq + [self.node_to_id[end_node]]
        
        # Determine max length (spec says 64, but 32 is sufficient for <20 tokens)
        max_len = 32
        
        x = torch.full((max_len,), self.special_tokens['[PAD]'], dtype=torch.long)
        y = torch.full((max_len,), -100, dtype=torch.long) # -100 is ignore_index
        
        # Fill x
        seq_len = len(full_seq)
        x[:seq_len-1] = torch.tensor(full_seq[:-1], dtype=torch.long)
        
        # Fill y (target is next token)
        # We only want to predict the very last token [C]
        # In a standard autoregressive setup:
        # Input at t: x[t]
        # Target at t: x[t+1]
        # We want loss ONLY when predicting the final answer.
        # The sequence fed to model is full_seq[:-1].
        # The target for position -1 is full_seq[-1].
        
        # Let's align with the user spec "Only compute loss on target token".
        # If input is "A->B. B->C. Q A ? ->", next token is "C".
        # So at the last position of input, target is C.
        
        target_pos = len(seq) - 1 # Position of the final '->' 
        
        # IMPORTANT: standard GPT forward(x, y) expects x and y to be same shape (B, T).
        # y is typically x shifted by 1.
        # Here we manually construct y to be -100 everywhere except the final position.
        
        # Standard:
        # x: [A, ->, B, ..., ->]
        # y_true: [->, B, ..., C]
        
        # We want y to be [-100, -100, ..., C]
        
        # Construct actual tokens
        actual_tokens = torch.tensor(full_seq, dtype=torch.long)
        length = len(actual_tokens)
        
        x[:length-1] = actual_tokens[:-1]
        # y should be shifted by 1
        y_shifted = actual_tokens[1:].clone()
        
        # Mask everything in y except the final token?
        # Spec: "Only compute loss on target token (final answer position)"
        # The final answer C is at index length-1 in actual_tokens.
        # In y_shifted (which corresponds to predictions from x[:length-1]), C is at index length-2.
        
        mask = torch.ones_like(y_shifted) * -100
        mask[-1] = y_shifted[-1] # The last token in the sequence (the answer)
        
        y[:length-1] = mask
        
        return x, y
