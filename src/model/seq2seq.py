import sys
from typing import Tuple, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Seq2Seq(nn.Module):
    def __init__(
        self,
        src_tok_emb,
        tgt_tok_emb,
        positional_encoding,
        transformer,
        generator,
        device,
        pad_token_id,
        bos_token_id,
        eos_token_id,
        **kwargs
    ):
        super(Seq2Seq, self).__init__()
        self.transformer = transformer
        self.src_tok_emb = src_tok_emb
        self.tgt_tok_emb = tgt_tok_emb
        self.positional_encoding = positional_encoding
        self.generator = generator
        self.device = device
        self.pad_token = pad_token_id
        self.bos_token = bos_token_id
        self.eos_token = eos_token_id

    def forward(self, src: Tensor, tgt: Tensor):
        # src: L x B
        # tgt: (L-1) x B
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self._create_mask(
            src, tgt
        )
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )
        # return = L x B x V
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )

    def translate(
        self,
        x,
        buffer: float = 0.0,
        max_length: int = sys.maxsize,
        context_size=sys.maxsize,
    ):
        in_L, B = x.shape
        in_fin = torch.zeros(B, dtype=torch.bool, device=self.device)
        in_all_fin_idx = sys.maxsize
        out_fin = torch.zeros(B, dtype=torch.bool, device=self.device)
        out_L = min(max_length, int(in_L * (buffer + 1)))
        output = torch.empty((out_L, B), dtype=torch.long, device=self.device)
        output[0] = self.bos_token
        for i in range(1, out_L):
            if i >= int(in_all_fin_idx * (buffer + 1)):
                break
            probs = F.softmax(
                self.forward(x, output[i - out_L - context_size : i])[-1], dim=-1
            )
            output[i] = torch.multinomial(probs, num_samples=1).transpose(0, 1)
            out_fin |= output[i] == self.eos_token
            if out_fin.all():
                i += 1
                break
            if i < in_L:
                in_fin |= x[i] == self.eos_token
                if in_all_fin_idx == sys.maxsize and in_fin.all():
                    in_all_fin_idx = i
        return output[:i]

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _create_mask(self, src: Tensor, tgt: Tensor):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(
            torch.float
        )
        src_padding_mask = (src == self.pad_token).transpose(0, 1).type(torch.float)
        tgt_padding_mask = (tgt == self.pad_token).transpose(0, 1).type(torch.float)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
