# JKO-iFlow

Official implementation of "Invertible normalizing flow neural networks by JKO scheme". Please direct inquiries to cxu310@gatech.edu.

## Pre-requisites
```
pip install -r requirements.txt
```

## Usage

We have simplified the code to make it minimally dependent on external packages and self-contained.

Run the codes below to train 2d flow on the non-trivial examples of rose and fractal trees (see Figure 3). 

* Rose:
```
python main.py --JKO_config configs/JKO_rose.yaml
```
* Fractal tree:
```
python main.py --JKO_config configs/JKO_tree.yaml
```

## Citation
```
@inproceedings{
    xu2023normalizing,
    title={Normalizing flow neural networks by {JKO} scheme},
    author={Chen Xu and Xiuyuan Cheng and Yao Xie},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=ZQMlfNijY5}
}
```
## Animation 
We show below the forward (data to noise) and backward (noise to data) process over time.
<p align="center">
  <img src="https://github.com/hamrel-cxu/JKO-iFlow/blob/main/animation.gif" width="800" height="450" />
</p>

