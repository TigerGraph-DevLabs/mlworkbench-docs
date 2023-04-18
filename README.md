# Tutorials for TigerGraph ML Workbench

**Update**
This repo is deprecated. Please refer to https://github.com/tigergraph/graph-ml-notebooks for more recent docs.

**Note**: The `main` branch is under active development and might contain breaking changes since the latest release. For docs most relavant to a specific version of the ML Workbench, please check out the corresponding branch, e.g., the `1.0` branch for ML Workbench v1.0.x.

## Repository Organization and Content

The repository is broken down into __tutorials__ and __demos__. The __tutorials__ directory contains basic notebooks to get you up and running with the TigerGraph ML Workbench, including basic data loading and manipulation functions, building GNN models, and then finally deploying your models to cloud providers such as Azure and Google Cloud Platform. The __demos__ directory contains notebooks that apply the basic concepts in the tutorials to real-world data and business problems.

### Tutorials
The tutorial directory is broken down to three subdirectories: `basics`, `gnn_pyg`, `cloud_deployment`, and `advanced`. The `basics` directory contains notebooks that are designed to get developers familiar with the different utilities that `pyTigerGraph` and the `TigerGraph Machine Learning Workbench` provide. Moving from that, the `gnn_pyg` directory contains notebooks that combine the data loading utilities introduced in the basics directory and uses them to train Graph Neural Networks built using PyTorch Geometric. The `cloud_deployment` directory contains notebooks that apply the concepts in the tutorials to deploy GNN models to cloud providers such as Azure and Google Cloud Platform. Finally, the `advanced` directory contains relativaly complex uses of ML workbench such as hyperparameter tuning. 

### Demos
There are currently two demos available: `ethereum_fraud` and `recommendation`. `etheruem_fraud` walks you through training a Graph Neural Network to detect fraudulent Ethereum accounts. `recommendation` walks you through training a Graph Neural Network to recommend music to users in the `LastFM` dataset.

## Table of Contents
