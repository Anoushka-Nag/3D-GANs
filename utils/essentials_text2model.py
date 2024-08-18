import torch
import torch.nn as nn
import torch.utils.data as data


def train(
        model: nn.Module,
        data_loader: data.DataLoader,
        criterion,
        optimizer,
        log_interval: int,
        device: torch.device = torch.device("cpu")
):
    model.train()

    running_loss = 0.0

    for batch_idx, data_point in enumerate(data_loader):
        batch_idx = batch_idx + 1

        x, y = data_point
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(f'Status: Batches [{batch_idx+1}/{len(data_loader)}] Loss: {round(loss.item(), 6)}')

    return running_loss / len(data_loader)
