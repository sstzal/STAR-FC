# STAR-FC #
This code is the implementation for the proposed STAR-FC. 

## Requirements ##

 - Python = 3.6 
 - Pytorch = 1.2.0
 - faiss
 - More details please refer to `requirements.txt`

## Hardware ##


- 24G TITAN RTX
- 48 core Intel Xeon CPU E5-2650-v4@2.20GHz processor

## Datasets ##

    cd STAR-FC

Create a new folder for training data:

    mkdir data

To run the code, please download the refined MS1M dataset and partition it into 10 splits, then construct the data directory as follows: (Since the supplementary material is limited to less than 100MB, we can not submit these data.)

    |——data
       |——features
          |——part0_train.bin
          |——part1_test.bin
          |——...
    	  |——part9_test.bin
        ——labels
          |——part0_train.meta
          |——part1_test.meta
          |——...
          |——part9_test.meta
        ——knns
          |——part0_train/faiss_k_80.npz
          |——part1_test/faiss_k_80.npz
          |——...
          |——part9_test/faiss_k_80.npz

## Model ##
The pretrained model `Backbone.pth` and `Head.pth` is available in `./pretrained_model`.

## Training ##

Adjust the configuration in `./src/configs/cfg_gcn_ms1m.py`, then run the algorithm as follows:

    cd STAR-FC
    sh scripts/train_gcn_ms1m.sh


## Testing ##

Adjust the configuration in `./src/configs/cfg_gcn_ms1m.py`, then run the algorithm as follows:

    cd STAR-FC
    python test_final.py