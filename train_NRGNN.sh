ptb_rate=0.2
for seed in {10..15};
do
    python train_NRGNN.py \
        --dataset cora \
        --seed ${seed} \
        --t_small 0.1 \
        --alpha 0.03\
        --lr 0.001 \
        --epochs 200 \
        --n_p -1 \
        --p_u 0.8 \
        --label_rate 0.05 \
        --ptb_rate ${ptb_rate} \
        --noise uniform


    python train_NRGNN.py \
        --dataset cora \
        --seed ${seed} \
        --t_small 0.1 \
        --alpha 0.3\
        --lr 0.001 \
        --epochs 200 \
        --n_p -1 \
        --p_u 0.8 \
        --label_rate 0.05 \
        --ptb_rate ${ptb_rate} \
        --noise pair
done


# python train_NRGNN.py \
#     --dataset citeseer \
#     --seed 14 \
#     --t_small 0.1 \
#     --alpha 3\
#     --lr 0.001 \
#     --epochs 200 \
#     --n_p -1 \
#     --p_u 0.8 \
#     --label_rate 0.05 \
#     --ptb_rate ${ptb_rate} \
#     --noise uniform

# python train_NRGNN.py \
#     --dataset citeseer \
#     --seed 14 \
#     --t_small 0.1 \
#     --alpha 3\
#     --lr 0.001 \
#     --epochs 200 \
#     --n_p -1 \
#     --p_u 0.7 \
#     --label_rate 0.05 \
#     --ptb_rate ${ptb_rate} \
#     --noise pair



# python train_NRGNN.py \
#     --dataset dblp \
#     --seed 11 \
#     --t_small 0.1 \
#     --alpha 0.003\
#     --lr 0.001 \
#     --epochs 200 \
#     --n_p 50 \
#     --p_u 0.7 \
#     --label_rate 0.01 \
#     --ptb_rate ${ptb_rate} \
#     --noise uniform


# python train_NRGNN.py \
#     --dataset dblp \
#     --seed 11 \
#     --t_small 0.1 \
#     --alpha 0.003\
#     --lr 0.001 \
#     --epochs 200 \
#     --n_p 50 \
#     --p_u 0.7 \
#     --label_rate 0.01 \
#     --ptb_rate ${ptb_rate} \
#     --noise pair

# python train_NRGNN.py \
#     --dataset pubmed \
#     --seed 10 \
#     --t_small 0.1 \
#     --alpha 0.1\
#     --lr 0.01 \
#     --epochs 200 \
#     --n_p 5 \
#     --p_u 1.0 \
#     --label_rate 0.01 \
#     --ptb_rate ${ptb_rate} \
#     --hidden 128 \
#     --noise uniform

# python train_NRGNN.py \
#     --dataset pubmed \
#     --seed 10 \
#     --t_small 0.1 \
#     --alpha 0.001\
#     --lr 0.01 \
#     --epochs 200 \
#     --n_p 5 \
#     --p_u 1.0 \
#     --label_rate 0.01 \
#     --ptb_rate ${ptb_rate} \
#     --hidden 128 \
#     --noise pair


