import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.diffusion_model import Diffusion, SmallUNet

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BATCH = 128
LR = 2e-4

device = "cpu"
torch.manual_seed(42)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_data = datasets.CIFAR10(root="data", train=True, download=True,
                              transform=transform)
loader = DataLoader(train_data, batch_size=BATCH, shuffle=True,
                    num_workers=0)

model = SmallUNet()
diffusion = Diffusion(model, timesteps=1000)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        t = torch.randint(0, diffusion.timesteps, (x.size(0),),
                          device=device)
        noise = torch.randn_like(x)
        x_t = diffusion.q_sample(x, t, noise)
        pred = model(x_t, t)
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    path = f"app/cifar_diffusion_ep{epoch}.pt"
    torch.save(model.state_dict(), path)
    print(f"epoch {epoch}: loss {total_loss / len(loader):.4f} "
          f"({time.time() - start:.0f}s) saved {path}", flush=True)

torch.save(model.state_dict(), "app/cifar_diffusion.pt")
print("done: app/cifar_diffusion.pt")
