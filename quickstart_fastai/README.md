---
tags: [quickstart]
dataset: [MNIST]
framework: [flower]
---

# Federated Learning with Flower on CloudLab (Quickstart Example)
This repository demonstrates running a federated learning experiment using the Flower framework with a [fastai](https://www.fast.ai/) CNN model on CloudLab nodes. Deep knowledge of fastai is not necessarily required to run the example. The example will help you understand how to adapt Flower to your specific use case, and running it is quite straightforward.

The setup consists of:
- **1 Server node** (coordinating training)
- **Multiple Client nodes** (each training on local data partitions)

## Prerequisites
- CloudLab account with reserved nodes.
- SSH key pair configured and uploaded to CloudLab portal.
- VS Code (or terminal) with SSH access configured.

## 1. Login to a CloudLab Node
SSH into your assigned node from VS Code (example for clnode289):
```bash
  ssh -i ~/.ssh/id_ed25519_cloudlab -vvv Parthaw@clnode289.clemson.cloudlab.us
```

## 2. Clone the Repository
```bash
  git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/quickstart-fastai . \
		&& rm -rf _tmp && cd quickstart-fastai
```

## 3. Setup Python Virtual Environment
```bash
  # install system package (use the version matching your python if needed)
  sudo apt update
  sudo apt install -y python3-venv

  # Create virtual environment
  python3 -m venv .venv

  # Activate/Accessing virtual environment
  source .venv/bin/activate
```
## 4. Install Dependencies:
Install the dependencies defined in pyproject.toml:
```bash
  pip install -e .
```

## 5. Run Local Scripts (Optional Check):
Before starting server-client execution, verify that scripts run correctly:
```bash
  python ./fastai_example/task.py
  python ./fastai_example/client_app.py
  python ./fastai_example/server.py
```

## 6. Run Federated Learning with Flower:
### On the Server Node (e.g., clnode289)
```bash
  python3 ./fastai_example/server_app.py --host 0.0.0.0 --port 8080 --rounds 5
```
### On the Client Nodes 
Client 1 (e.g., clnode293)
```bash
  python3 ./fastai_example/client_app.py --server-address clnode289.clemson.cloudlab.us:8080 --partition-id 0
```
Client 2 (e.g., clnode287)
```bash
  python3 ./fastai_example/client_app.py --server-address clnode289.clemson.cloudlab.us:8080 --partition-id 1
```
Client 3 (e.g., clnode302)
```bash
  python3 ./fastai_example/client_app.py --server-address clnode289.clemson.cloudlab.us:8080 --partition-id 2
```

## Notes
- Replace clnodeXXX.clemson.cloudlab.us with the actual hostnames of your reserved nodes.
- Ensure that the server is running before starting the clients.
- Each client uses a different --partition-id for distinct data partitions.

## References
- [Flower Framework](https://flower.ai/)
- [FastAI](https://docs.fast.ai/)
- [CloudLab](https://www.cloudlab.us/)
