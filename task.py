# ...existing code...
"""fastai_example: A Flower / Fastai app."""

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Resize, ToTensor

# New imports
import torchvision.models as models
import torch.nn as nn
import numpy as np
from typing import List, Tuple

fds = None  # Cache FederatedDataset


def load_data(
    partition_id: int,
    num_partitions: int = 3,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # Ensure partitioner uses integer number of partitions (clients)
        partitioner = IidPartitioner(num_partitions=int(num_partitions))
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
            trust_remote_code=True,
        )
    # Validate partition id
    if int(partition_id) < 0 or int(partition_id) >= int(num_partitions):
        raise ValueError(
            f"partition_id must be in [0, {int(num_partitions)-1}] but got {partition_id}"
        )

    partition = fds.load_partition(int(partition_id), "train")

    # Resize and repeat channels to use MNIST, which have grayscale images,
    # with squeezenet, which expects 3 channels.
    # Ref: https://discuss.pytorch.org/t/fine-tuning-squeezenet-for-mnist-dataset/31221/2
    pytorch_transforms = Compose(
        [Resize(224), ToTensor(), Lambda(lambda x: x.expand(3, -1, -1))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    def collate_fn(batch):
        """Change the dictionary to tuple to keep the exact dataloader behavior."""
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)

        return images_tensor, labels_tensor

    partition = partition.with_transform(apply_transforms)
    # 20 % for on federated evaluation
    partition_full = partition.train_test_split(test_size=0.2, seed=42)
    # 60 % for the federated train and 20 % for the federated validation (both in fit)
    partition_train_valid = partition_full["train"].train_test_split(
        train_size=0.75, seed=42
    )
    trainloader = DataLoader(
        partition_train_valid["train"],
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valloader = DataLoader(
        partition_train_valid["test"],
        batch_size=32,
        collate_fn=collate_fn,
    )
    testloader = DataLoader(
        partition_full["test"], batch_size=32, collate_fn=collate_fn, num_workers=1
    )
    return trainloader, valloader, testloader

# ...existing code...
# Added utilities: model creation, parameter conversion, train/eval helpers


def build_model(num_classes: int = 10) -> nn.Module:
    """Create the model used for federated training (squeezenet adapted)."""
    model = models.squeezenet1_1(pretrained=False, num_classes=num_classes)
    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Return model parameters as list of numpy ndarrays (order matches state_dict)."""
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Set model parameters from a list of numpy ndarrays (order must match state_dict)."""
    state_dict_keys = list(model.state_dict().keys())
    if len(state_dict_keys) != len(parameters):
        raise ValueError(
            f"Mismatch between model state_dict keys ({len(state_dict_keys)}) and parameters ({len(parameters)})"
        )
    new_state_dict = {}
    for k, arr in zip(state_dict_keys, parameters):
        tensor = torch.from_numpy(arr)
        new_state_dict[k] = tensor
    model.load_state_dict(new_state_dict)


def train_local(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int = 1,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> None:
    """Train model on local data for `epochs`."""
    if device is None:
        device = get_device()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()


def evaluate_local(
    model: nn.Module, dataloader: DataLoader, device: torch.device | None = None
) -> Tuple[float, int]:
    """Evaluate model on given dataloader. Returns (loss, num_examples)."""
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss_sum += float(criterion(preds, yb))
            total += yb.size(0)
            pred_labels = preds.argmax(dim=1)
            correct += int((pred_labels == yb).sum())
    avg_loss = loss_sum / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, total