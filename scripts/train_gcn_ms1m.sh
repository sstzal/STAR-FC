cfg_name=cfg_gcn_ms1m
config=src/configs/$cfg_name.py

export PYTHONPATH=.

# train
python src/main.py \
    --config $config \
    --phase 'train'

