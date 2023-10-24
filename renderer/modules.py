import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import LFQ

class GLUResBlock(nn.Module):
    def __init__(self, chan: int, groups: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan * 2, 3, padding=1),
            nn.GLU(dim=1),
            nn.GroupNorm(groups, chan),
            nn.Conv2d(chan, chan * 2, 3, padding=1),
            nn.GLU(dim=1),
            nn.GroupNorm(groups, chan),
        )

    def forward(self, x):
        return self.net(x) + x


class ResBlock(nn.Module):
    def __init__(
        self, chan: int, groups: int = 16, act: nn.Module = nn.LeakyReLU, **act_kwargs
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.GroupNorm(groups, chan),
            act(**act_kwargs),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.GroupNorm(groups, chan),
            act(**act_kwargs),
        )

    def forward(self, x):
        return self.net(x) + x


class Encoder(nn.Module):
    def __init__(
        self,
        in_chan: int = 3,
        first_chan: int = 16,
        n_layers: int = 4,
        act: nn.Module = nn.LeakyReLU,
        **act_kwargs
    ):
        super().__init__()

        self.net = nn.ModuleList(
            [nn.Conv2d(in_chan, first_chan, 5, 1, 2), nn.LeakyReLU(0.1)]
        )

        for i in range(n_layers):
            self.net.append(
                nn.Sequential(
                    nn.Conv2d(
                        first_chan * (2**i), first_chan * (2 ** (i + 1)), 4, 2, 1
                    ),
                    act(**act_kwargs),
                    GLUResBlock(first_chan * (2 ** (i + 1))),
                )
            )

    def forward(self, x):
        for l in self.net:
            x = l(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        out_chan: int = 3,
        last_chan: int = 16,
        n_layers: int = 4,
        upsample_mode: str = "nearest",
        act: nn.Module = nn.LeakyReLU,
        **act_kwargs
    ):
        super().__init__()

        self.net = nn.ModuleList(
            [
                nn.Conv2d(last_chan, out_chan, 1),
            ]
        )

        for i in range(n_layers):
            self.net.append(
                nn.Sequential(
                    nn.Upsample(mode=upsample_mode, scale_factor=2),
                    nn.Conv2d(
                        last_chan * (2 ** (i + 1)), last_chan * (2**i), 3, 1, 1
                    ),
                    nn.LeakyReLU(0.1),
                    ResBlock(last_chan * (2**i), act=act, **act_kwargs),
                )
            )

        self.net = self.net[::-1]

    def forward(self, x):
        for l in self.net:
            x = l(x)
        return x


class VQVAE(nn.Module):
    def __init__(
        self,
        chan: int = 3,
        first_chan: int = 16,
        n_layers: int = 4,
        codebook_args: dict = {
            "codebook_size": 2**17,
        },
        quantizer: nn.Module = LFQ,
        **kwargs
    ):
        super().__init__()
        enc_kwargs = {
            key.split("encoder_")[1]: value
            for key, value in kwargs.items()
            if key.startswith("encoder")
        }
        dec_kwargs = {
            key.split("decoder_")[1]: value
            for key, value in kwargs.items()
            if key.startswith("decoder")
        }

        self.encoder = Encoder(chan, first_chan, n_layers, **enc_kwargs)
        self.quantizer = quantizer(**codebook_args, dim=first_chan * (2**n_layers))
        self.decoder = Decoder(chan, first_chan, n_layers, **dec_kwargs)

    def encode(self, x):
        return self.encoder(x)

    def quantize(self, x):
        return self.quantizer(x)

    def decode(self, x):
        return self.decode(x)

    def forward(self, x):
        x = self.encoder(x)
        x, idx, loss = self.quantizer(x)
        x = self.decoder(x)
        return x, idx, loss


class Discriminator(nn.Module):
    def __init__(
        self,
        chan: int = 3,
        first_chan: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()

        self.net = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(chan, first_chan, 5, padding=2), nn.LeakyReLU())]
        )

        for i in range(n_layers):
            self.net.append(
                nn.Sequential(
                    nn.Conv2d(
                        first_chan * (2**i), first_chan * (2 ** (i + 1)), 4, 2, 1
                    ),
                    nn.GroupNorm(16, first_chan * (2 ** (i + 1))),
                    nn.LeakyReLU(0.1),
                )
            )

        self.net.append(nn.Conv2d(first_chan * (2**n_layers), 1, 4))

    def forward(self, x):
        for net in self.net:
            x = net(x)
        return x


class Predictor(nn.Module):
    def __init__(self, n_emojis: int = 4099, embed_dim: int = 60, out_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(n_emojis, embed_dim)

        self.linear = nn.Sequential(
            nn.Linear(embed_dim + 4, out_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(out_dim // 2, out_dim),
        )

    def forward(self, emoji_idx, pos):
        embeddings = self.embed(emoji_idx)
        x = torch.cat([embeddings, pos], dim=-1)
        return self.linear(x)

def hinge_loss(real, fake):
    return torch.mean(F.relu(1 - real)) + torch.mean(F.relu(1 + fake))