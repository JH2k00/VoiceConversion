import torch
from torch import nn
from torch.nn import functional as F

from vector_quantize_pytorch import VectorQuantize, FSQ, ResidualVQ

# Borrowed from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_channel):
        super().__init__()
        self.blocks = nn.Sequential(*[
            nn.Conv2d(in_channel, channel, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
            ResBlock(channel, n_res_channel),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
            ResBlock(channel, n_res_channel),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, 4, stride=(1,2), padding=1),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=(1,2), padding=1),
            ])
        
    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_channel, n_speakers):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel), 
            nn.ConvTranspose2d(channel, channel, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(channel),
            nn.ConvTranspose2d(channel, channel, 4, stride=(1, 2), padding=1),
            nn.BatchNorm2d(channel),
            ResBlock(channel, n_res_channel),
            nn.BatchNorm2d(channel),
            nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            ResBlock(channel, n_res_channel),
            nn.BatchNorm2d(channel),
            nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1),
        ])
        
        self.embedding_layers = nn.ModuleList([nn.Sequential(nn.Embedding(n_speakers, channel), nn.BatchNorm1d(channel)) for _ in range(8)])

    def forward(self, input, speaker_onehot):
        x = input
        embedding_layer_idx = 0
        for layer in self.blocks:
            x = layer(x)
            if(isinstance(layer, nn.BatchNorm2d)):
                speaker_embedding = self.embedding_layers[embedding_layer_idx](speaker_onehot)[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
                x = F.silu(x + speaker_embedding, inplace=True)
                embedding_layer_idx += 1
        return x

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_channel=32,
        embed_dim=256,
        n_embed=512,
        n_speakers=128,
        decay=0.8,
    ):
        super().__init__()

        self.enc = Encoder(in_channel, channel, n_res_channel)
        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)

        # self.quantize = VectorQuantize(embed_dim, n_embed, decay=decay)
        self.quantize = FSQ(levels=[8, 5, 5, 5])

        self.dec = Decoder(embed_dim, 1, channel, n_res_channel, n_speakers)

    def forward(self, input, speaker_id):
        quantized = self.encode(input)
        return self.decode(quantized, speaker_id)

    def encode(self, input):
        enc = self.quantize_conv(self.enc(input))
        enc, _, = self.quantize(enc.permute(0, 2, 3, 1)) # quantized, indices
        return enc.permute(0, 3, 1, 2)

    def decode(self, quant, speaker_id):
        return self.dec(quant, speaker_id)