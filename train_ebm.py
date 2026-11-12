import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.ebm_model import EnergyNet, langevin_sample

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
BATCH = 64
LR = 1e-4
BUFFER_SIZE = 10000
LANGEVIN_STEPS = 40
STEP_SIZE = 0.1

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_data = datasets.CIFAR10(root="data", train=True, download=True,
                              transform=transform)
loader = DataLoader(train_data, batch_size=BATCH, shuffle=True,
                    num_workers=0, drop_last=True)

model = EnergyNet()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
buffer = torch.rand(BUFFER_SIZE, 3, 32, 32) * 2 - 1

CKPT = "app/cifar_ebm_ckpt.pt"
start_it = 1
if os.path.exists(CKPT):
    ckpt = torch.load(CKPT)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    buffer = ckpt["buffer"]
    start_it = ckpt["iter"] + 1
    print(f"resumed from iter {ckpt['iter']}", flush=True)

data_iter = iter(loader)
start = time.time()
for it in range(start_it, ITERS + 1):
    try:
        real, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        real, _ = next(data_iter)

    idx = torch.randint(0, BUFFER_SIZE, (BATCH,))
    fake_init = buffer[idx]
    fresh = torch.rand(BATCH, 3, 32, 32) * 2 - 1
    from_buffer = (torch.rand(BATCH) < 0.95)[:, None, None, None]
    fake_init = torch.where(from_buffer, fake_init, fresh)
    fake = langevin_sample(model, fake_init, steps=LANGEVIN_STEPS,
                           step_size=STEP_SIZE)

    e_real = model(real)
    e_fake = model(fake)
    loss = (e_real - e_fake).mean() + \
        0.1 * (e_real ** 2 + e_fake ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    buffer[idx] = fake.detach()

    if it % 100 == 0:
        print(f"iter {it}: loss {loss.item():.3f} "
              f"E_real {e_real.mean().item():.3f} "
              f"E_fake {e_fake.mean().item():.3f} "
              f"({time.time() - start:.0f}s)", flush=True)
    if it % 100 == 0:
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "buffer": buffer, "iter": it}, CKPT)
        print(f"saved {CKPT}", flush=True)

torch.save(model.state_dict(), "app/cifar_ebm.pt")
print("done: app/cifar_ebm.pt")
