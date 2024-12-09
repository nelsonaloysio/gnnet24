# Deep Community Detection in Attributed Temporal Graphs: Experimental Evaluation of Current Approaches (ACM CoNEXT/GNNet'24)

This is the code repository to reproduce the results from the paper:

> Passos, N.A.R.A., Carlini, E., Trani, S. (2024). [Deep Community Detection in Attributed Temporal Graphs: Experimental Evaluation of Current Approaches](https://doi.org/10.1145/3694811.3697822). In Proceedings of the 3rd Graph Neural Networking Workshop 2024 (GNNet '24). Association for Computing Machinery, New York, NY, USA, 1–6.

> **To-do: update TGC model implementation for dataset compatibility (see [original repository](https://github.com/MGitHubL/Deep-Temporal-Graph-Clustering)).**

___

This repository tree follows the structure below:

* `code`: Code implementation for models in our experiment (see [Models](#models) below).
* `data`: Graph datasets used in our papers in numpy format (see [Datasets](#datasets) below).
* `params`: Hyperparameters used for each model in our experiments, including seed numbers.

[Models](code/gnnet24/models) have a CLI with default arguments set to the hyperparameters we used.

### Requires

Requirements can be installed from [pypi (requirements.txt)](requirements.txt) or using [conda (environment.yml)](environment.yml).

### Usage

To reproduce all clustering results:

```bash
bash run_experiments.sh arxivai brain dblp patent pubmed school
```

Or just those using the **Leiden** algorithm on **DBLP** dataset:

```bash
python run.py dblp pubmed --seed 2354 4512 5694 6614 8745
```

## Models

Models receive node-level features, static graphs, snapshot-based temporal graphs (STG) or event-based temporal graphs (ETG) as input:

| | Input | Topology | Features | Temporal |
| :---: | :---: | :---: | :---: | :---: |
| **K-Means** | Features | | ✓ | |
| **Spectral Clustering** | Graph | ✓ | | |
| **[Leiden](https://doi.org/10.1038/s41598-019-41695-z)** | Graph | ✓ | | |
| **[Node2Vec](https://doi.org/10.1145/2939672.2939754)** | Graph | ✓ | | |
| **[DynNode2Vec](https://doi.org/10.1109/BigData.2018.8621910)** | STG | ✓ | | ✓ |
| **[tNodeEmbed](https://doi.org/10.5555/3367471.3367683)** | STG | ✓ | | ✓ |
| **[DAEGC](https://doi.org/10.5555/3367471.3367551)** | Graph | ✓ | ✓ | |
| **[TGC](https://openreview.net/pdf?id=ViNe1fjGME)** | ETG | ✓ | ✓ | ✓ |

## Datasets

The number of edges (static), interactions (temporal), and components (disjoint subgraphs) consider an undirected graph loaded by PyG with self-loops and node isolates removed:

| | Nodes | Edges | Interactions | Components | Features | Classes | Time steps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[arXivAI](https://doi.org/10.48550/arXiv.2306.04962)** | 69,854 | 696,819 | 696,819 | 244 | 128 | 5 | 27 |
| **[Brain](https://doi.org/10.1016/j.neuroimage.2016.12.061)** | 5,000 | 878,207 | 947,744 | 1 | 128 | 10 | 12 |
| **[DBLP](https://doi.org/10.1145/3219819.3220054)** | 28,085 | 150,568 | 222,165 | 113 | 128 | 10 | 27 |
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
