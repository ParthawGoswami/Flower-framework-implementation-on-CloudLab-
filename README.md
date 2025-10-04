Federated training with a simple HTTP aggregator

This example runs a minimal federated learning workflow using an HTTP aggregator.

Files
- server_app.py: Flask aggregator that accepts client weight submissions, averages them each round, and exposes global weights.
- client_app.py: Client runner that downloads weights, trains locally on a partition, and uploads updated weights.
- task.py: Data loading utilities (uses flwr_datasets to partition MNIST).

Quick start (on CloudLab)

1. On the server node (e.g. clnode289):

   python3 -m pip install -r requirements.txt
   python3 server_app.py --port 5000 --num-clients 3 --num-rounds 3

2. On each client node (replace partition-id and client-id):

   python3 -m pip install -r requirements.txt
   python3 client_app.py --server http://clnode289.clemson.cloudlab.us:5000 --client-id client1 --partition-id 0 --num-partitions 3 --num-rounds 3

Notes
- Make sure the server's hostname (or IP) is reachable from client nodes and any firewall rules allow the port.
- This is a simple proof-of-concept aggregator; it doesn't implement authentication or robust failure handling.
- For debugging, increase logging or add print statements.
