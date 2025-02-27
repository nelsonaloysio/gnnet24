# acm-conext-gnnet24

This is the code repository to reproduce the results from the paper:

> Passos, N.A.R.A., Carlini, E., Trani, S. (2024). [Deep Community Detection in Attributed Temporal Graphs: Experimental Evaluation of Current Approaches](https://doi.org/10.1145/3694811.3697822). In Proceedings of the 3rd Graph Neural Networking Workshop 2024 (GNNet '24). Association for Computing Machinery, New York, NY, USA, 1–6.

___

## Usage

Each model available in `code/models` has a command line interface (CLI), in which the default argument values are set to the hyperparameters used for our experimental evaluation.

To make thing easier, a unified CLI is also available in `code/run.py`. For example, to reproduce the results obtained with the **Leiden** algorithm on the **PubMed** dataset:

```bash
./run.py leiden --data pubmed --seed 2354 4512 5694 6614 8745
```

Alternatively, all models may be sequentially trained on a specific dataset by running `code/run.sh $DATASET_NAME`.

> **Note:** all experiments were run with the same set of seeds above. For details on each model and dataset, please refer to the description below and their code implementations.

___

## Description

In this paper, an experimental evaluation of 9 models for node-level clustering or community detection was performed on 6 datasets. This repository tree follows the structure below:

* `code`: Code implementation for each model we experimented with (see [Models](#models) below).
* `data`: Graph datasets available in compressed ZIP format (see [Datasets](#datasets) below).

### Models

| | Input | Topology | Features | Temporal |
| :---: | :---: | :---: | :---: | :---: |
| **K-Means** | $X_V$​ | | ✓ | |
| **Spectral Clustering** | $G$ | ✓ | | |
| **Leiden** | $G$ | ✓ | | |
| **Node2Vec** | $G$ | ✓ | | |
| **DynNode2Vec** | $G_S​$ | ✓ | | ✓ |
| **tNodeEmbed** | $G_S​$ | ✓ | | ✓ |
| **DAEGC** | $G$ | ✓ | ✓ | |
| **TGC** | $G_E​$ | ✓ | ✓ | ✓ |

> **Note:** input refers to node-level features only ($X$), static graphs ($G$), snapshot-based ($G_S$) temporal graphs or event-based ($G_E$) temporal graphs.

### Datasets

| | Nodes | Edges | Interactions | Components | Features | Classes | Time steps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **arXivAI** | 69,854 | 696,819 | 696,819 | 244 | 128 | 5 | 27 |
| **Brain** | 5,000 | 883,207 | 1,007,744 | 1 | 128 | 10 | 12 |
| **DBLP** | 28,085 | 150,571 | 222,169 | 113 | 128 | 10 | 27 |
| **Patent** | 12,214 | 41,916 | 41,916 | 5 | 128 | 6 | 891 |
| **PubMed** | 19,717 | 44,324 | 44,324 | 1 | 500 | 3 | 42 |
| **School** | 327 | 5,818 | 188,508 | 1 | 128 | 9 | 7,375 |

Both the number of edges and interactions above consider an undirected graph.

Additional datasets not included in our paper are also available in the `data` folder.

> **Note:** With the exception of [PubMed](https://github.com/nelsonaloysio/pubmed-temporal), node-level features for the datasets were obtained with Node2Vec by the authors of TGC (see: [this repository](https://duckduckgo.com/?q=DuckDuckGo+AI+Chat&ia=chat&duckai=1)).

___

## Cite

```
@inproceedings{10.1145/3694811.3697822,
    author = {Reis de Almeida Passos, Nelson Aloysio and Carlini, Emanuele and Trani, Salvatore},
    title = {Deep Community Detection in Attributed Temporal Graphs: Experimental Evaluation of Current Approaches},
    year = {2024},
    isbn = {9798400712548},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3694811.3697822},
    doi = {10.1145/3694811.3697822},
    booktitle = {Proceedings of the 3rd Graph Neural Networking Workshop 2024},
    pages = {1--6},
    numpages = {6},
    keywords = {graph neural networks, ndoe clustering, temporal graphs},
    location = {Los Angeles, United States of America},
    series = {GNNet '24},
    abstract = {Recent advances in network representation learning have sparked renewed interest in developing strategies for learning on spatio-temporal signals, crucial for applications like traffic forecasting, recommendation systems, and social network analysis. Despite the popularity of Graph Neural Networks for node-level clustering, most specialized solutions are evaluated in transductive learning settings, where the entire graph is available during training, leaving a significant gap in understanding their performance in inductive learning settings. This work presents an experimental evaluation of community detection approaches on temporal graphs, comparing traditional methods with deep learning models geared toward node-level clustering. We assess their performance on six real-world datasets, focused on a transductive setting and extending to an inductive setting for one dataset. Our results show that deep learning models for graphs do not consistently outperform more established methods on this task, highlighting the need for more effective approaches and comprehensive benchmarks for their evaluation.},
}
```
