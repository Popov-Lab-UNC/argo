# gen_models/gem/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import logging

from . import utils

# --- PyTorch Sub-modules for the Transformer ---

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head, self.d_k, self.d_v = n_head, d_k, d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_k, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, _ = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, None

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device), diagonal=1).unsqueeze(0).byte()
    return subsequent_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_inner, pad_idx=None, dropout=0.1, n_position=200):
        super().__init__()
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_word_vec, d_inner, n_head, d_k, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_word_vec, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))
        enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, slf_attn_mask=src_mask)
        return enc_output

class Decoder(nn.Module):
    def __init__(self, n_src_vocab, d_word_vec):
        super().__init__()
        self.decoder = nn.Linear(d_word_vec, n_src_vocab)

    def forward(self, enc_output):
        return self.decoder(enc_output)

# --- The Main Transformer Model ---

class Transformer(nn.Module):
    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_inner, pad_idx=None, dropout=0.1, n_position=200,
                 start_token='<', end_token='>', pad_token='X', max_len=120):
        super().__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.max_len = max_len
        self.pad_idx = utils.CHAR_2_IDX[self.pad_token]

        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_inner, pad_idx, dropout, n_position)
        self.decoder = Decoder(n_src_vocab, d_word_vec)

    def forward(self, seq, return_attns=False):
        mask = get_subsequent_mask(seq)
        enc_output = self.encoder(seq, mask, return_attns=return_attns)
        return self.decoder(enc_output)

    def fit(self, loader, optimizer, scheduler, n_epochs, device):
        self.to(device)
        self.train()
        for epoch in range(n_epochs):
            for seq in loader:
                seq = seq.to(device, dtype=torch.int64)
                optimizer.zero_grad()

                pad_mask = (seq[:, :-1] != self.pad_idx).float().view(-1)
                logits = self(seq)[:, :-1].contiguous().view(-1, len(utils.TOKENS))
                labels = seq[:, 1:].contiguous().view(-1)

                criterion = nn.CrossEntropyLoss(reduction='none')
                loss = criterion(logits, labels)
                
                masked_loss = (loss * pad_mask).sum() / pad_mask.sum()
                masked_loss.backward()
                optimizer.step()
            
            if scheduler:
                scheduler.step()

    def generate(self, batch_size, device):
        self.to(device)
        self.eval()

        start_idx = utils.CHAR_2_IDX[self.start_token]
        end_idx = utils.CHAR_2_IDX[self.end_token]
        
        seq = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(self.max_len):
                logits = self(seq)[:, -1]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop generation for sequences that have ended
                finished_mask = (seq[:, -1] == end_idx) | (seq[:, -1] == self.pad_idx)
                next_token[finished_mask] = self.pad_idx

                seq = torch.cat([seq, next_token], dim=1)
                if (seq == self.pad_idx).all(): # All sequences finished
                    break
        
        generated_smiles = []
        for i in range(batch_size):
            token_ids = seq[i].cpu().numpy()
            # Trim at start/end/pad tokens
            try:
                start_pos = np.where(token_ids == start_idx)[0][0] + 1
                end_pos = np.where(token_ids == end_idx)[0][0]
                smiles = "".join(utils.TOKENS[t] for t in token_ids[start_pos:end_pos])
                generated_smiles.append(smiles)
            except IndexError:
                # Handle cases where start/end token is not found
                generated_smiles.append("") 
        
        return generated_smiles

# --- Dataset for the Transformer ---
class SmilesDataset(Dataset):
    def __init__(self, smiles: list, max_len=120, start_token='<', end_token='>', pad_token='X'):
        self.smiles = smiles
        self.max_len = max_len
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        smi = self.start_token + smi + self.end_token
        padded_smi = smi.ljust(self.max_len + 2, self.pad_token)[:self.max_len + 2]
        return torch.tensor([utils.CHAR_2_IDX[c] for c in padded_smi], dtype=torch.long)

class GEM:
    def __init__(self, model_path: str, device=None):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.load_pretrained_transformer(model_path, self.device)

    @staticmethod
    def load_pretrained_transformer(model_path: str, device):
        n_src_vocab = len(utils.TOKENS)
        model_params = {
            'n_src_vocab': n_src_vocab, 'd_word_vec': 512, 'n_layers': 8,
            'n_head': 8, 'd_k': 64, 'd_inner': 1024
        }
        model = Transformer(**model_params)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        logging.info(f"Loaded pre-trained Transformer model from {model_path}")
        return model

    def fine_tune(self, smiles_data: list, lr: float = 1e-5, n_epochs: int = 10, save_path: str = None):
        logging.info(f"Starting fine-tuning for {n_epochs} epochs with {len(smiles_data)} SMILES.")
        dataset = SmilesDataset(smiles_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.fit(loader, optimizer, scheduler=None, n_epochs=n_epochs, device=self.device)
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            logging.info(f"Fine-tuned model saved to {save_path}")
        return self.model

    def generate(self, n_samples: int = 100, n_trials: int = 1):
        total = n_samples * n_trials
        all_generated = []
        for _ in range(n_trials):
            batch = self.model.generate(n_samples, self.device)
            all_generated.extend(batch)
        return all_generated
    
    def save_checkpoint(self, save_path: str):
        torch.save(self.model.state_dict(), save_path)
        logging.info(f"Model checkpoint saved to {save_path}")
        return