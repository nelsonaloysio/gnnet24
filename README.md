# Deep Community Detection in Attributed Temporal Graphs: Experimental Evaluation of Current Approaches (ACM CoNEXT/GNNet'24)

This is the code repository to reproduce the results from the paper:

> Passos, N.A.R.A., Carlini, E., Trani, S. (2024). [Deep Community Detection in Attributed Temporal Graphs: Experimental Evaluation of Current Approaches](https://doi.org/10.1145/3694811.3697822). In Proceedings of the 3rd Graph Neural Networking Workshop 2024 (GNNet '24). Association for Computing Machinery, New York, NY, USA, 1–6.

## Requirements

Tested with **Python 3.8**. To install requirements:

```bash
pip install -r requirements.txt
```

___

## Usage

Each model available in `code/models` has a command line interface (CLI), with default argument values set to the hyperparameters used in our experimental evaluation.

To make thing easier, a unified CLI is also available. For example, to reproduce the results obtained with the **Leiden** algorithm on the **DBLP** dataset:

```bash
python run.py dblp pubmed --seed 2354 4512 5694 6614 8745
```

> **Note:** All experiments were run with the same set of seeds above.

Passing multiple arguments in the command line will permutate them (i.e., perform a grid search).

For more information on models and datasets, please refer to the [Description](#description) section below.

### Reproduce clustering results

A shell script is included that allows to quickly reproduce our obtained results:

```bash
bash run_experiments.sh
```

> **Note:** To retrain each model from scratch instead, add `--retrain` to the command line above.

___

## Description

In this paper, an experimental evaluation of 9 models for node-level clustering or community detection was performed on 6 datasets. This repository tree follows the structure below:

* `code/models`: Code implementation for each model we experimented with (see [Models](#models) below).
* `code/utils`: Utility functions for early stop, evaluation, and loading and preprocessing data.
* `data`: Graph datasets used in our papers, in NumPy format (see [Datasets](#datasets) below).
* `params`: Hyperparameters used in our experiments for each model, including seed numbers.

### Models

Models receive node-level features ($X$), static graphs ($G$), snapshot-based temporal graphs ($G_S$) or event-based temporal graphs ($G_E$), as summarized in the table below:

| | Input | Topology | Features | Temporal |
| :---: | :---: | :---: | :---: | :---: |
| **K-Means** | $X_V$​ | | ✓ | |
| **Spectral Clustering** | $G$ | ✓ | | |
| **[Leiden](https://doi.org/10.1038/s41598-019-41695-z)** | $G$ | ✓ | | |
| **[Node2Vec](https://doi.org/10.1145/2939672.2939754)** | $G$ | ✓ | | |
| **[DynNode2Vec](https://doi.org/10.1109/BigData.2018.8621910)** | $G_S​$ | ✓ | | ✓ |
| **[tNodeEmbed](https://doi.org/10.5555/3367471.3367683)** | $G_S​$ | ✓ | | ✓ |
| **[DAEGC](https://doi.org/10.5555/3367471.3367551)** | $G$ | ✓ | ✓ | |
| **[TGC](https://openreview.net/pdf?id=ViNe1fjGME)** | $G_E​$ | ✓ | ✓ | ✓ |

### Datasets

The number of edges (static), interactions (temporal), and components (disjoint subgraphs) below consider an undirected graph matching the dataset loaded by PyTorch Geometric:

| | Nodes | Edges | Interactions | Components | Features | Classes | Time steps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[arXivAI](https://doi.org/10.48550/arXiv.2306.04962)** | 69,854 | 696,819 | 696,819 | 244 | 128 | 5 | 27 |
| **[Brain](https://doi.org/10.1016/j.neuroimage.2016.12.061)** | 5,000 | 883,207 | 1,007,744 | 1 | 128 | 10 | 12 |
| **[DBLP](https://doi.org/10.1145/3219819.3220054)** | 28,085 | 150,571 | 222,169 | 113 | 128 | 10 | 27 |
| **[Patent](https://doi.org/10.3386/w8498)** | 12,214 | 41,916 | 41,916 | 5 | 128 | 6 | 891 |
| **[PubMed](https://zenodo.org/records/13932075)** | 19,717 | 44,324 | 44,324 | 1 | 500 | 3 | 42 |
| **[School](https://doi.org/10.1371/journal.pone.0136497)** | 327 | 5,818 | 188,508 | 1 | 128 | 9 | 7,375 |

> **Note:** With the exception of [PubMed](https://github.com/nelsonaloysio/pubmed-temporal), node-level features for the datasets were obtained with Node2Vec, following the approach outlined by the authors of TGC (see: [Deep-Temporal-Graph-Clustering@MGitHubL](https://github.com/MGitHubL/Deep-Temporal-Graph-Clustering)).

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
