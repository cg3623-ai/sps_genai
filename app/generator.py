import base64
import io
from pathlib import Path

import torch
from PIL import Image

from app.gan_model import Generator


class DigitGenerator:
    def __init__(self, weights_path=None, latent_dim=100):
        if weights_path is None:
            weights_path = Path(__file__).parent / 'mnist_gan.pt'
        self.latent_dim = latent_dim
        self.model = Generator(latent_dim)
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.model.eval()

    def generate(self, seed=None) -> dict:
        if seed is not None:
            torch.manual_seed(seed)
        z = torch.randn(1, self.latent_dim)
        with torch.no_grad():
            image = self.model(z)[0, 0]
        pixels = ((image + 1) / 2 * 255).byte().numpy()
        buffer = io.BytesIO()
        Image.fromarray(pixels, mode='L').save(buffer, format='PNG')
        return {
            'image_base64': base64.b64encode(buffer.getvalue()).decode(),
            'seed': seed,
            'format': 'png',
            'size': '28x28',
        }
