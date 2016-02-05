# refer_google_mmi
This code is the re-implementation of [Generation and Comprehension of Unambiguous Object Descriptions](http://arxiv.org/abs/1511.02283)

## Download
Download my cleaned data and extract them into "./data" folder
- 1) http://tlberg.cs.unc.edu/licheng/referit/data/refclef.zip
- 2) http://tlberg.cs.unc.edu/licheng/referit/data/refcoco.zip
- 3) http://tlberg.cs.unc.edu/licheng/referit/data/refcoco+.zip 

## Prepare Images:
Besides we add "images/mscoco" into the "./data" folder. 
Download images from [mscoco](http://mscoco.org/dataset/#overview)

## Download my pretrained model:
I release my trained baseline (model_id0) and mmi models (model_id10) for refcoco_licheng here, extract them into "./model" folder: http://tlberg.cs.unc.edu/licheng/referit/model/refcoco_licheng.zip

## Prerequisites
To use the code, you need to have skimage, h5py, scipy installed for python, and cunn, cudnn, cutorch, nn, nngraph, image, loadcaffe installed for Torch7.

## How to train
Firstly, we need to prepare data.json and data.h5 for each dataset_splitBy, e.g., refcoco_licheng, refcoco_google, etc.
```bash
$ python prepro.py --dataset refcoco --splitBy licheng --max_length 10
```

Next, we could call train.lua to learn the baseline model by setting ranking_weight as 0. That said, the ranking loss won't be effectively back-propagated.
```bash
$ th train.lua -dataset refcoco_licheng -ranking_weight 0
```

Or we can call train.lua to learn the Max-Mutual Information model if you set ranking_weight greater than 0.
```bash
$ th train.lua -dataset refcoco_licheng -ranking_weight 2
```

The above two calls only learn the jemb (joint_embedding) and LSTM parameters.
Besides, if you want to finetune CNN as well, you can call
```bash
$ th train.lua -dataset refcoco_licheng -ranking_weight 2 -cnn_finetune 1
```
Also you could change "-finetune_jemb_after 20000" to specify after how many iterations would CNN finetuning begin.


## How to test
call eval_lang.lua to compute BLEU, METEOR, and CIDER scores
```bash
$ th eval_lang.lua -dataset refcoco_licheng -split testA
```
It will also call vis_lang.py and write visualization into cache/vis/dataset_splitBy/

call eval_box.lua to compute sent->box accuracy
```bash
$ th eval_box.lua -dataset refcoco_licheng -split testA
```
Note we have testA and testB in refcoco_licheng, but only test in refcoco_google. It's due to different style of split. 
