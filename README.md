# Cluster_Analyzer [![arXiv](https://img.shields.io/badge/arXiv-2412.00568---?logo=arXiv&labelColor=b31b1b&color=grey)](https://arxiv.org/abs/2411.07189)

`cluster_analyzer` is a Python library designed to analyze critical behavior, self-similarity, and fractal dimensions of clusters in cellular automata. It provides tools for simulating cluster formation, performing statistical analyses, and visualizing results.

<p align="center">
    <img src="https://raw.githubusercontent.com/HakanAkgn/ClusterAnalyzer/main/assets/Feature_Image.png" width="800" />
</p>

## Features
- Optimized cluster detection
- Simulate Logistic Game of Life (LGOL) and find polynomial solutions.
- Analyze cluster sizes and their distributions.
- Perform power-law fitting and goodness-of-fit tests and visualizations based on [powerlaw package](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0085777) developed in on the statistical methods developed in [Clauset et al. 2007](https://arxiv.org/abs/0706.1062) and [Klaus et al. 2011](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0019779).
- Generate Probability Density Function (PDF) and Complementary Cumulative Distribution Function (CCDF) plots.

## Installation

You can install ClusterAnalyzer via pip:

```bash
$ pip install cluster-analyzer
```

or clone the repository and install it manually:
```bash
$ git clone https://github.com/HakanAkgn/ClusterAnalyzer.git
$ cd ClusterAnalyzer
$ pip install . 
```
to install optional dependencies to reproduce all the figures in the paper, run:
```bash
$ pip install .[all]
```

Check the installation:
```python
import cluster_analyzer as ca
print(ca.__version__)
```

## Usage

See the [Demo Notebook](https://github.com/HakanAkgn/ClusterAnalyzer/blob/main/ClusterAnalyzerDemo.ipynb) for examples of each function.

See the [Data Display Notebook](https://github.com/HakanAkgn/ClusterAnalyzer/blob/main/Paper_Data/Data_Display.ipynb) for a detailed view of the dataset and visualizations used in the paper.

## Citation
If you find this work useful, please cite our paper:

```bibtex
@article{akgun2026deterministic,
  title={Deterministic scale-invariant dynamics in a logistic Game-of-Life model},
  author={Akg{\"u}n, Hakan and Yan, Xianquan and Ta{\c{s}}k{\i}ran, Tamer and Ibrahimi, Muhamet and Lee, Ching Hua and Jahangirov, Seymur},
  journal={Communications Physics},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```