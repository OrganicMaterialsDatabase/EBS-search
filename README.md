# Online Search Tool for Graphical Patterns in Electronic Band Structures

This repository hosts the code and a test case for the paper [arXiv:1710.11611](https://arxiv.org/abs/1710.11611). Tested with Python 3.6.5.

To see the tool in action, register for free at: [omdb.diracmaterials.org](https://omdb.diracmaterials.org)

## Installation
```
pip install -r requirements.txt
```

## Example
First, create an ANN (approximate nearest neighbours) index:
```
python create_index.py --dataset folder --band_index 0 --width 0.4 --dimensions 16 --trees 10
```
