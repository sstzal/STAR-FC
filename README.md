# STAR-FC #
This code is the implementation for the CVPR 2021 paper "Structure-Aware Face Clustering on a Large-Scale Graph with 10^7 Nodes". 

## Requirements ##

 - Python = 3.6 
 - Pytorch = 1.2.0
 - faiss

## Hardware ##


- 24G TITAN RTX
- 48 core Intel Xeon CPU E5-2650-v4@2.20GHz processor

## Datasets ##

    cd STAR-FC

Create a new folder for training data:

    mkdir data

To run the code, please download the refined MS1M dataset and partition it into 10 splits, then construct the data directory as follows:

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
 We have used the data from: [https://github.com/yl-1993/learn-to-cluster](https://github.com/yl-1993/learn-to-cluster)

## Model ##
Put the pretrained model `Backbone.pth` and `Head.pth` in the `./pretrained_model`.

## Training ##

Adjust the configuration in `./src/configs/cfg_gcn_ms1m.py`, then run the algorithm as follows:

    cd STAR-FC
    sh scripts/train_gcn_ms1m.sh


## Testing ##

Adjust the configuration in `./src/configs/cfg_gcn_ms1m.py`, then run the algorithm as follows:

    cd STAR-FC
    python test_final.py
