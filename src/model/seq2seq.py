import sys
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
        # src: Length x Batch
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
        # return = L x B x Vocab
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
        self, x, buffer: float = 0.5, max_length: int = sys.maxsize, context_size=None
    ):
        output = (
            torch.ones(1, x.shape[1])
            .fill_(self.bos_token)
            .type(torch.long)
            .to(self.device)
        )
        in_fin = torch.zeros(x.shape[1]).bool().to(self.device)
        in_all_fin_idx = sys.maxsize
        out_fin = torch.zeros(x.shape[1]).bool().to(self.device)
        for i in range(max_length):
            if in_fin.all() and in_all_fin_idx == sys.maxsize:
                in_all_fin_idx = i
            if out_fin.all() or i > in_all_fin_idx * (buffer + 1):
                break
            context = output[:, -context_size:] if context_size is not None else output
            logits = self.forward(x, context)
            logits = logits[-1, :, :]
            probs = F.softmax(logits, dim=-1)
            output_next = torch.multinomial(probs, num_samples=1).transpose(0, 1)
            output = torch.cat([output, output_next], dim=0)
            out_fin |= output_next.squeeze() == self.eos_token
            if i < x.shape[0]:
                in_fin |= x[i].squeeze() == self.eos_token
        return output

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
