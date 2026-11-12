import base64
import io
from pathlib import Path

import torch
from PIL import Image

from app.diffusion_model import Diffusion, SmallUNet


class DiffusionGenerator:
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = Path(__file__).parent / 'cifar_diffusion.pt'
        self.model = SmallUNet()
        self.model.load_state_dict(
            torch.load(weights_path, map_location='cpu'))
        self.diffusion = Diffusion(self.model, timesteps=1000)

    def generate(self, seed=None) -> dict:
        if seed is not None:
            torch.manual_seed(seed)
        image = self.diffusion.sample(n=1, steps=100)[0]
        pixels = ((image + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
        buffer = io.BytesIO()
        Image.fromarray(pixels, mode='RGB').save(buffer, format='PNG')
        return {
            'image_base64': base64.b64encode(buffer.getvalue()).decode(),
            'seed': seed,
            'format': 'png',
            'size': '32x32',
        }
