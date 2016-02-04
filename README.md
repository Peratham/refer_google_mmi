Full implementation of google paper. More detailed description to be added soon.
# refer_google_mmi

## Download
Download my cleaned data from:
- 1) http://tlberg.cs.unc.edu/licheng/referit/data/refclef.zip
- 2) http://tlberg.cs.unc.edu/licheng/referit/data/refcoco.zip
- 3) http://tlberg.cs.unc.edu/licheng/referit/data/refcoco+.zip 
and extract them into "./data" folder

## Prepare Images:
Besides we add "images/mscoco" into the "./data" folder. 
Download images from [mscoco](http://mscoco.org/dataset/#overview)

## Download my pretrained model:
to be uploaded soon

## How to train
call prepro.py to make data.json and data.h5
```bash
$ python prepro.py -dataset refcoco -splitBy licheng -max_length 10
```
call train.lua to learn the baseline model
```bash
$ th train.lua -dataset refcoco_licheng -ranking_weight 0
```

call train.lua to learn the Max-Mutual Information model
```bash
$ th train.lua -dataset refcoco_licheng -ranking_weight 2
```
You need to make sure the ranking_weight is greater than 0.

If you want to finetune CNN as well, call
```bash
$ th train.lua -dataset refcoco_licheng -ranking_weight 2 -cnn_finetune 1
```
Also you could change the number of "finetune_jemb_after" to specify the number of iterations before finetuning CNN+joint_embedding layers.

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
