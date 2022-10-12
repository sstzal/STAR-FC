cfg_name=cfg_gcn_ms1m_hierarchical
config=src/configs/$cfg_name.py

export PYTHONPATH=.

# train
python src/main.py \
    --config $config \
    --phase 'train' \
    --k_num2 $1 \
    --k_num3 $2
