import argparse
import logging

import torch
import torch.nn.functional as F
from pyTigerGraph import TigerGraphConnection
from pyTigerGraph.gds.metrics import Accuracy
from torch_geometric.nn import GraphSAGE


def train(model, data, optimizer, epochs, device):
    data = data.to(device)
    for epoch in range(epochs):
        # Train
        model.train()
        acc = Accuracy()
        # Forward pass
        out = model(data.x.float(), data.edge_index)
        # Calculate loss
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Evaluate
        val_acc = Accuracy()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc.update(pred[data.train_mask], data.y[data.train_mask])
            valid_loss = F.cross_entropy(
                out[data.val_mask], data.y[data.val_mask])
            val_acc.update(pred[data.val_mask], data.y[data.val_mask])
        # Logging
        logging.info("train_accuracy={:.4f}".format(acc.value))
        logging.info("valid_accuracy={:.4f}".format(val_acc.value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--embed",
        type=int,
        default=64,
        metavar="EMB",
        help="embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        metavar="L",
        help="number of graph convolution layers (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        metavar="D",
        help="dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.001,
        metavar="L2",
        help="weight of the l2 panelty term (default: .001)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.DEBUG,
    )

    conn = TigerGraphConnection(
        host="http://127.0.0.1", # Change the address to your database server's
        graphname="Cora",
        username="tigergraph",
        password="tigergraph",
        gsqlSecret="", # secret instead of user/pass is required for TG cloud DBs created after 7/5/2022
    )
    # Uncomment below if token authentication is enabled on the DB
    # conn.getToken("YOUR SECRET") # Change to your DB secret

    graph_loader = conn.gds.graphLoader(
        v_in_feats=["x"],
        v_out_labels=["y"],
        v_extra_feats=["train_mask", "val_mask", "test_mask"],
        num_batches=1,
        output_format="PyG",
        shuffle=False,
    )

    data = graph_loader.data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphSAGE(
        in_channels=1433,
        hidden_channels=args.embed,
        num_layers=args.layers,
        out_channels=7,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2)

    train(model, data, optimizer, args.epochs, device)


if __name__ == "__main__":
    main()
