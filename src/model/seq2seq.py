import torch
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
        pad_token,
        **kwargs
    ):
        super(Seq2Seq, self).__init__()
        self.transformer = transformer
        self.src_tok_emb = src_tok_emb
        self.tgt_tok_emb = tgt_tok_emb
        self.positional_encoding = positional_encoding
        self.generator = generator
        self.device = device
        self.pad_token = pad_token

    def forward(self, src: Tensor, tgt: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
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
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(
            torch.bool
        )

        src_padding_mask = (src == self.pad_token).transpose(0, 1)
        tgt_padding_mask = (tgt == self.pad_token).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
