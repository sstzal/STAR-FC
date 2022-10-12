cfg_name=cfg_gcn_ms1m
config=src/configs/$cfg_name.py

export PYTHONPATH=.

# train
python src/main.py \
    --config $config \
    --phase 'train_hierarchical' \
    --k_num2 $1 \
    --k_num3 $2
