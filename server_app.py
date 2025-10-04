import argparse
import flwr as fl
import torch
from task import build_model, get_parameters
import numpy as np
from typing import Dict, List, Tuple

def weighted_accuracy_aggregation(
    results: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
    """Aggregate per-client accuracy into a weighted average by number of examples."""
    total_examples = sum(num_examples for num_examples, _ in results)
    if total_examples == 0:
        return {"accuracy": 0.0}
    acc_sum = 0.0
    for num_examples, metrics in results:
        acc = float(metrics.get("accuracy", 0.0))
        acc_sum += acc * num_examples
    return {"accuracy": acc_sum / total_examples}

def main():
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for Flower server")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds")
    args = parser.parse_args()

    server_address = f"{args.host}:{args.port}"

    # Build initial model and extract parameters to seed global model
    model = build_model()
    initial_weights = get_parameters(model)

    # Convert to Flower Parameters object
    initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)

    # Create FedAvg strategy with initial parameters
    strategy = fl.server.strategy.FedAvg(initial_parameters=initial_parameters, evaluate_metrics_aggregation_fn=weighted_accuracy_aggregation)

    # Start Flower server
    fl.server.start_server(server_address=server_address, config=fl.server.ServerConfig(num_rounds=args.rounds), strategy=strategy)


if __name__ == "__main__":
    main()
