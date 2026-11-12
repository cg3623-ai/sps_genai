import base64
import io
from pathlib import Path

import torch
from PIL import Image

from app.ebm_model import EnergyNet, langevin_sample


class EBMGenerator:
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = Path(__file__).parent / 'cifar_ebm.pt'
        self.model = EnergyNet()
        self.model.load_state_dict(
            torch.load(weights_path, map_location='cpu'))
        self.model.eval()

    def generate(self, seed=None) -> dict:
        if seed is not None:
            torch.manual_seed(seed)
        start = torch.rand(1, 3, 32, 32) * 2 - 1
        image = langevin_sample(self.model, start, steps=250,
                                step_size=0.1)[0]
        pixels = ((image + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
        buffer = io.BytesIO()
        Image.fromarray(pixels, mode='RGB').save(buffer, format='PNG')
        return {
            'image_base64': base64.b64encode(buffer.getvalue()).decode(),
            'seed': seed,
            'format': 'png',
            'size': '32x32',
        }
