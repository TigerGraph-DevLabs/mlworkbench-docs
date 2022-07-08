# Ethereum Fraud Demo

Use a Graph Neural Network to detect fraud in an Ethereum transaction network.

First, setup a TigerGraph database instance.

If you don't have a Python environment with the appropriate dependencies, you can install them with the following command:

```sh
conda env create -f environment.yml
```

Then run the `schema_load_preprocess.ipynb` notebook.

Finally, run the `gnn_fraud_detection.ipynb` notebook.