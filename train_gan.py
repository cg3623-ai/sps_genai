import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from app.gan_model import Discriminator, Generator

LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.0002

torch.manual_seed(42)
device = torch.device('cpu')


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True,
                             transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, LATENT_DIM)
    os.makedirs('samples', exist_ok=True)

    start_epoch = 0
    ckpt_path = 'app/checkpoint_latest.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        opt_g.load_state_dict(ckpt['opt_g'])
        opt_d.load_state_dict(ckpt['opt_d'])
        start_epoch = ckpt['epoch']
        print(f'Resuming from epoch {start_epoch}', flush=True)
    elif os.path.exists('app/mnist_gan_ep5.pt'):
        generator.load_state_dict(torch.load('app/mnist_gan_ep5.pt',
                                             map_location=device))
        start_epoch = 5
        print('Loaded epoch-5 generator weights, continuing from epoch 5',
              flush=True)

    for epoch in range(start_epoch, EPOCHS):
        for i, (real_images, _) in enumerate(loader):
            batch = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = torch.ones(batch, 1)
            fake_labels = torch.zeros(batch, 1)

            opt_d.zero_grad()
            loss_real = criterion(discriminator(real_images), real_labels)
            noise = torch.randn(batch, LATENT_DIM)
            fake_images = generator(noise)
            loss_fake = criterion(discriminator(fake_images.detach()), fake_labels)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            loss_g = criterion(discriminator(fake_images), real_labels)
            loss_g.backward()
            opt_g.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{EPOCHS}] Step [{i}/{len(loader)}] '
                      f'Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}',
                      flush=True)

        with torch.no_grad():
            samples = generator(fixed_noise)
        utils.save_image(samples, f'samples/epoch_{epoch + 1:02d}.png',
                         nrow=8, normalize=True)
        torch.save({'epoch': epoch + 1,
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'opt_g': opt_g.state_dict(),
                    'opt_d': opt_d.state_dict()}, ckpt_path)
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            torch.save(generator.state_dict(), f'app/mnist_gan_ep{epoch + 1}.pt')
            print(f'Saved samples/epoch_{epoch + 1:02d}.png', flush=True)

    torch.save(generator.state_dict(), 'app/mnist_gan.pt')
    print('Done. Final generator weights saved to app/mnist_gan.pt')


if __name__ == '__main__':
    main()
