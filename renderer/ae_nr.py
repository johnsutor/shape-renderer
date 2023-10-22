import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.InstanceNorm2d(dim_out),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        h = self.block2(h)

        return h + self.res_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(in_c, in_c, 3, 1, 1)
        )

    def forward(self, x):
        return self.up(x)


class Encoder(nn.Module):
    def __init__(
        self, img_size: int = 256, first_channels: int = 16, out_dim: int = 512
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            ResnetBlock(3, first_channels),
            nn.Conv2d(first_channels, first_channels, 4, 2, 1),
            ResnetBlock(first_channels, first_channels * 2),
            nn.Conv2d(first_channels * 2, first_channels * 2, 4, 2, 1),
            ResnetBlock(first_channels * 2, first_channels * 4),
            nn.Conv2d(first_channels * 4, first_channels * 4, 4, 2, 1),
            ResnetBlock(first_channels * 4, first_channels * 8),
            nn.Conv2d(first_channels * 8, first_channels * 8, 4, 2, 1),
        )
        self.linear = nn.Sequential(
            nn.Linear(8 * img_size * first_channels, 2 * out_dim),
            nn.SiLU(),
            nn.Linear(2 * out_dim, out_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        return self.linear(x)


class Decoder(nn.Module):
    def __init__(self, in_dim: int = 512, last_channels: int = 16, img_size: int = 256):
        super().__init__()
        self.last_channels = last_channels

        self.linear = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.SiLU(),
            nn.Linear(in_dim * 2, 8 * img_size * last_channels),
            nn.SiLU(),
        )

        self.backbone = nn.Sequential(
            UpBlock(last_channels * 8),
            ResnetBlock(last_channels * 8, last_channels * 4),
            UpBlock(last_channels * 4),
            ResnetBlock(last_channels * 4, last_channels * 2),
            UpBlock(last_channels * 2),
            ResnetBlock(last_channels * 2, last_channels),
            UpBlock(last_channels),
            nn.Conv2d(last_channels, 3, 1, 1),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(
            x.shape[0], self.last_channels * 8, self.last_channels, self.last_channels
        )
        x = self.backbone(x)
        return x


class Predictor(nn.Module):
    def __init__(self, n_emojis: int = 4099, embed_dim: int = 60, out_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(n_emojis, embed_dim)

        self.linear = nn.Sequential(
            self.block(embed_dim + 4, out_dim // 8),
            self.block(out_dim // 8, out_dim // 4),
            self.block(out_dim // 4, out_dim // 2),
            nn.Linear(out_dim // 2, out_dim),
        )

    def block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, emoji_idx, pos):
        embeddings = self.embed(emoji_idx)
        x = torch.cat([embeddings, pos], dim=-1)
        return self.linear(x)


class Renderer(nn.Module):
    def __init__(self, img_size: int = 256):
        super().__init__()
        self.encoder = Encoder(img_size=img_size)
        self.decoder = Decoder(img_size=img_size)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec
