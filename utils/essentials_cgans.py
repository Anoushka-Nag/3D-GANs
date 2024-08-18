import torch
import torch.nn as nn
import torch.utils.data as data


def train_discriminator(
        discriminator: nn.Module,
        generator: nn.Module,
        data_loader: data.DataLoader,
        noise_generator,
        criterion_discriminator,
        optimizer_discriminator,
        log_interval: int,
        device: torch.device = torch.device("cpu")
):
    discriminator.train()
    generator.train()

    running_loss = 0.0

    for batch_idx, (real_x, real_out) in enumerate(data_loader):
        batch_idx = batch_idx + 1

        batch_size = real_x.shape[0]

        real_x = real_x.to(device)
        real_out = real_out.to(device)
        real_y = torch.ones((batch_size, 1)).to(device)

        fake_noise = noise_generator(batch_size).to(device)
        fake_x = torch.empty_like(real_x).copy_(real_x).to(device)
        fake_out = generator(fake_x, fake_noise)
        fake_y = torch.zeros((batch_size, 1)).to(device)

        optimizer_discriminator.zero_grad()

        real_pred = discriminator(real_out, real_x)
        fake_pred = discriminator(fake_out, fake_x)

        real_loss = criterion_discriminator(real_pred, real_y)
        fake_loss = criterion_discriminator(fake_pred, fake_y)

        total_loss = real_loss + fake_loss
        total_loss.backward()

        optimizer_discriminator.step()

        running_loss += total_loss.item()

        if batch_idx % log_interval == 0:
            print(f'Status: Batches [{batch_idx}/{len(data_loader)}] Loss: {round(total_loss.item(), 6)}')

    return running_loss / len(data_loader)


def train_generator(
        generator: nn.Module,
        discriminator: nn.Module,
        data_loader: data.DataLoader,
        criterion_generator,
        optimizer_generator,
        log_interval: int,
        device: torch.device = torch.device("cpu")
):
    generator.train()
    discriminator.train()

    running_loss = 0.0
    for batch_idx, (fake_x, fake_noise) in enumerate(data_loader):
        batch_idx = batch_idx + 1

        batch_size = fake_x.shape[0]

        fake_x = fake_x.to(device)
        fake_noise = fake_noise.to(device)
        fake_y = torch.ones((batch_size, 1)).to(device)

        optimizer_generator.zero_grad()

        fake_out = generator(fake_x, fake_noise)
        fake_out = discriminator(fake_out, fake_x)
        fake_loss = criterion_generator(fake_out, fake_y)
        fake_loss.backward()

        optimizer_generator.step()

        running_loss += fake_loss.item()

        if batch_idx % log_interval == 0:
            print(f'Status: Batches [{batch_idx}/{len(data_loader)}] Loss: {round(fake_loss.item(), 6)}')

    return running_loss / len(data_loader)
