import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch \
            else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init_conv = nn.Conv2d(3, 32, 3, padding=1)
        self.down1 = ResBlock(32, 32, time_dim)
        self.pool1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.down2 = ResBlock(64, 64, time_dim)
        self.pool2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.mid = ResBlock(128, 128, time_dim)
        self.up1 = ResBlock(128 + 64, 64, time_dim)
        self.up2 = ResBlock(64 + 32, 32, time_dim)
        self.out_norm = nn.GroupNorm(8, 32)
        self.out_conv = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        x0 = self.init_conv(x)
        d1 = self.down1(x0, t_emb)
        d2 = self.down2(self.pool1(d1), t_emb)
        mid = self.mid(self.pool2(d2), t_emb)
        u1 = F.interpolate(mid, scale_factor=2, mode="nearest")
        u1 = self.up1(torch.cat([u1, d2], dim=1), t_emb)
        u2 = F.interpolate(u1, scale_factor=2, mode="nearest")
        u2 = self.up2(torch.cat([u2, d1], dim=1), t_emb)
        return self.out_conv(F.silu(self.out_norm(u2)))


class Diffusion:
    def __init__(self, model, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0, t, noise):
        a = self.alphas_cumprod.to(x0.device)[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, n=1, steps=100, device="cpu"):
        self.model.eval()
        grid = torch.linspace(0, self.timesteps - 1, steps).long()
        grid = torch.unique(grid)
        x = torch.randn(n, 3, 32, 32, device=device)
        ac = self.alphas_cumprod.to(device)
        for i in reversed(range(len(grid))):
            t = grid[i].item()
            t_tensor = torch.full((n,), t, device=device,
                                  dtype=torch.long)
            eps = self.model(x, t_tensor)
            a_bar = ac[t]
            x0_pred = (x - (1 - a_bar).sqrt() * eps) / a_bar.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            if i == 0:
                x = x0_pred
            else:
                a_prev = ac[grid[i - 1].item()]
                beta_tilde = (1 - a_prev) / (1 - a_bar) * \
                    (1 - a_bar / a_prev)
                mean = (a_prev.sqrt() * (1 - a_bar / a_prev) /
                        (1 - a_bar)) * x0_pred + \
                    ((a_bar / a_prev).sqrt() * (1 - a_prev) /
                     (1 - a_bar)) * x
                x = mean + beta_tilde.sqrt() * torch.randn_like(x)
        return x
