# Label Noise-Resistant Graph Neural Network

Offical pytorch implementation of proposed framework **NRGNN** and **Compared Methods** in "[NRGNN: Learning a Label Noise-Resistant Graph Neural Network on Sparsely and Noisily Labeled Graphs" (KDD 2021)](https://arxiv.org/pdf/2106.04714.pdf). If you find this repo to be useful, please cite our paper. Thank you.
```
@article{dai2021nrgnn,
  title={NRGNN: Learning a Label Noise-Resistant Graph Neural Network on Sparsely and Noisily Labeled Graphs},
  author={Dai, Enyan and Aggarwal, Charu and Wang, Suhang},
  journal={SIGKDD},
  year={2021}
}
```

## Content
- [Label Noise-Resistant Graph Neural Network](#label-noise-resistant-graph-neural-network)
  - [Content](#content)
  - [1. Requirements](#1-requirements)
  - [2. NRGNN](#2-nrgnn)
    - [2.1 Introduction](#21-introduction)
    - [Reproduc the Results](#reproduc-the-results)
  - [3. Compared Methods (to test)](#3-compared-methods-to-test)
    - [Co-teaching+](#co-teaching)
    - [D-GNN](#d-gnn)
    - [LafAK (CP)](#lafak-cp)
  - [4. Dataset](#4-dataset)

## 1. Requirements

```
torch==1.7.1
torch-geometric==1.7.1
```
The packages can be installed by directly run the commands in `install.sh` by
```
bash install.sh
```

## 2. NRGNN

### 2.1 Introduction
<div align=center><img src="https://github.com/EnyanDai/NRGNN/blob/main/Framework.png" width="700"/></div>

we propose to link the unlabeled nodes with labeled nodes of high feature similarity to bring more clean label information. Furthermore, accurate pseudo labels could be obtained by this strategy to provide more supervision and further reduce the effects of label noise.
<!-- ## Abstract 
Graph Neural Networks (GNNs) have achieved promising results for semi-supervised learning tasks on graphs such as node classification. Despite the great success of GNNs, many real-world graphs are often sparsely and noisily labeled, which could significantly degrade the performance of GNNs, as the noisy information could propagate to unlabeled nodes via graph structure. Thus, it is important to develop a label noise-resistant GNN for semi-supervised node classification. Though extensive studies have been conducted to learn neural networks with noisy labels, they mostly focus on independent and identically distributed data and assume a large number of noisy labels are available, which are not directly applicable for GNNs. Thus, we investigate a novel problem of learning a robust GNN with noisy and limited labels. To alleviate the negative effects of label noise,  Our theoretical and empirical analysis verify the effectiveness of these two strategies under mild conditions. Extensive experiments on real-world datasets demonstrate the effectiveness of the proposed method in learning a robust GNN with noisy and limited labels. -->



### Reproduc the Results

An example of training NRGNN:
```
python train_NRGNN.py \
    --dataset cora \
    --seed 11 \
    --t_small 0.1 \
    --alpha 0.03\
    --lr 0.001 \
    --epochs 200 \
    --n_p -1 \
    --p_u 0.8 \
    --label_rate 0.05 \
    --ptb_rate 0.2 \
    --noise uniform
```
All the hyper-parameters settings for the datasets are included in [`train_NRGNN.sh`](https://github.com/EnyanDai/NRGNN/tree/main/train_NRGNN.sh). To reproduce the performance reported in the paper, you can run the bash file:
```
bash train_NRGNN.sh
```
## 3. Compared Methods (to test)
### Co-teaching+
From Yu, Xingrui, et al. "How does disagreement help generalization against label corruption?." [[model](https://github.com/EnyanDai/NRGNN/blob/main/models/Coteaching.py), [trianing_example](https://github.com/EnyanDai/NRGNN/blob/main/baseline/train_Coteaching.py)]

### D-GNN
D-GNN is based on S-model from NT, Hoang, Choong Jun Jin, and Tsuyoshi Murata. "Learning graph neural networks with noisy labels." [[model](https://github.com/EnyanDai/NRGNN/blob/main/models/S_model.py), [trianing_example](https://github.com/EnyanDai/NRGNN/blob/main/baseline/train_S_model.py)]

### LafAK (CP)

From Zhang, Mengmei, et al. "Adversarial label-flipping attack and defense for graph neural networks." [[model](https://github.com/EnyanDai/NRGNN/blob/main/models/CP.py), [trianing_example](https://github.com/EnyanDai/NRGNN/blob/main/baseline/train_CP.py)] (Note that it is required to run deepwalk to obtain node embedding for the code.)

Our reimplemention of these methods are in `./models`. Their example training codes are in `./baseline`. 


## 4. Dataset
The DBLP can be automatically downloaded to `./data` through torch-geometric API. Other datasets are listed in `./data`





