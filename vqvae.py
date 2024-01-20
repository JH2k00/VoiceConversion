import torch
from torch import nn
from torch.nn import functional as F

from vector_quantize_pytorch import VectorQuantize
from wavenet_model import WaveNetModel

# Borrowed from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
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
            nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            ResBlock(channel // 2, n_res_channel),
            nn.BatchNorm2d(channel // 2),
            nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            ResBlock(channel // 2, n_res_channel),
            nn.BatchNorm2d(channel // 2),
            nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        ])

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_channel, n_speakers):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel), 
            nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ConvTranspose2d(channel // 2, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            ResBlock(channel // 2, n_res_channel),
            nn.ConvTranspose2d(channel // 2, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ConvTranspose2d(channel // 2, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            ResBlock(channel // 2, n_res_channel),
            nn.BatchNorm2d(channel // 2),
            nn.ConvTranspose2d(channel // 2, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
        ])
        
        self.embedding_layers = nn.ModuleList([nn.Linear(n_speakers, channel // 2) for _ in range(7)])

    def forward(self, input, speaker_onehot):
        x = input
        embedding_layer_idx = 0
        for layer in self.blocks:
            x = layer(x)
            if(layer is nn.BatchNorm1d):
                speaker_embedding = self.embedding_layers[embedding_layer_idx](speaker_onehot)[:, :, None].expand(-1, -1, x.shape[2], x.shape[3])
                x = F.relu(x + speaker_embedding, inplace=True)
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

        self.quantize = VectorQuantize(embed_dim, n_embed, decay=decay)

        self.dec = Decoder(embed_dim, 256, channel, n_res_channel, n_speakers)
        # self.dec = WaveNetModel(layers=9, output_length=131072)

    def forward(self, input, speaker_id):
        quantized, indices, vq_loss = self.encode(input)
        quantized = torch.permute(quantized, (0,2,1))
        return self.decode(quantized, speaker_id), vq_loss

    def encode(self, input):
        enc = self.quantize_conv(self.enc(input))
        print(enc.shape)
        enc = torch.permute(enc, (0,2,1))
        return self.quantize(enc) # quantized, indices, vq_loss

    def decode(self, quant, speaker_id):
        return self.dec(quant, speaker_id)