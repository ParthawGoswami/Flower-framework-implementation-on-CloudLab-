# ...existing code...
import argparse
import flwr as fl
import numpy as np
from task import (
    load_data,
    build_model,
    get_parameters,
    set_parameters,
    train_local,
    evaluate_local,
    get_device,
)
from typing import Dict, List, Tuple
import torch.nn as nn
import torch


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, partition_id: int, num_partitions: int = 3):
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.trainloader, self.valloader, self.testloader = load_data(
            partition_id, num_partitions
        )
        self.model: nn.Module = build_model()
        self.device = get_device()

    def get_parameters(self) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        # Receive global weights from server
        set_parameters(self.model, parameters)
        # Train locally. The server may pass epochs in config; default to 1 if absent.
        local_epochs = int(config.get("local_epochs", 1)) if config else 1
        lr = float(config.get("lr", 1e-3)) if config else 1e-3
        train_local(self.model, self.trainloader, epochs=local_epochs, lr=lr, device=self.device)
        # Return updated weights and number of training examples
        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        # Set received weights and evaluate on local validation/test set
        set_parameters(self.model, parameters)

        # Compute loss and accuracy on valloader
        device = self.device
        self.model.to(device)
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss_sum = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in self.valloader:
                xb, yb = xb.to(device), yb.to(device)
                preds = self.model(xb)
                loss_sum += float(criterion(preds, yb))
                total += yb.size(0)
                pred_labels = preds.argmax(dim=1)
                correct += int((pred_labels == yb).sum())

        avg_loss = loss_sum / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        # Return loss, number of examples, and metrics dict including accuracy
        return float(avg_loss), int(total), {"accuracy": float(accuracy)}


def main():
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
        help="Address of the Flower server (ip:port)",
    )
    parser.add_argument(
        "--partition-id", type=int, required=True, help="Partition id for this client (0..K-1)"
    )
    parser.add_argument("--num-partitions", type=int, default=3, help="Total number of partitions/clients")
    args = parser.parse_args()

    client = FlowerClient(partition_id=args.partition_id, num_partitions=args.num_partitions)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
# ...existing code...