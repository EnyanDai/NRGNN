# NRGNN 

A PyTorch implementation of "NRGNN: Learning a Label Noise-Resistant Graph Neural Network on Sparsely and Noisily Labeled Graphs" (KDD 2021). [[paper]](https://arxiv.org/pdf/2106.04714.pdf)


<div align=center><img src="https://github.com/EnyanDai/NRGNN/blob/main/Framework.png" width="700"/></div>

## Abstract 
Graph Neural Networks (GNNs) have achieved promising results for semi-supervised learning tasks on graphs such as node classification. Despite the great success of GNNs, many real-world graphs are often sparsely and noisily labeled, which could significantly degrade the performance of GNNs, as the noisy information could propagate to unlabeled nodes via graph structure. Thus, it is important to develop a label noise-resistant GNN for semi-supervised node classification. Though extensive studies have been conducted to learn neural networks with noisy labels, they mostly focus on independent and identically distributed data and assume a large number of noisy labels are available, which are not directly applicable for GNNs. Thus, we investigate a novel problem of learning a robust GNN with noisy and limited labels. To alleviate the negative effects of label noise, we propose to link the unlabeled nodes with labeled nodes of high feature similarity to bring more clean label information. Furthermore, accurate pseudo labels could be obtained by this strategy to provide more supervision and further reduce the effects of label noise. Our theoretical and empirical analysis verify the effectiveness of these two strategies under mild conditions. Extensive experiments on real-world datasets demonstrate the effectiveness of the proposed method in learning a robust GNN with noisy and limited labels.

## Requirements

```
torch==1.4.0
torch-geometric==1.6.1
```

## Run the code
After installation, you can clone this repository
```
git clone https://github.com/EnyanDai/NRGNN.git
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
## Dataset
The DBLP can be automatically downloaded to `./data` through torch-geometric API. Other datasets are listed in `./data`

## Reproduce the results
All the hyper-parameters settings for the datasets are included in [`train_NRGNN.sh`](https://github.com/EnyanDai/NRGNN/tree/main/train_NRGNN.sh).

To reproduce the performance reported in the paper, you can run the bash file.
```
bash train_NRGNN.sh
```


## Cite

If you find this repo to be useful, please cite our paper. Thank you.
```
@article{dai2021nrgnn,
  title={NRGNN: Learning a Label Noise-Resistant Graph Neural Network on Sparsely and Noisily Labeled Graphs},
  author={Dai, Enyan and Aggarwal, Charu and Wang, Suhang},
  journal={SIGKDD},
  year={2021}
}
```
