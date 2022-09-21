#!/bin/bash

# Prepare Yelp dataset
#if ! test -f "../data/yelp_review_full.train"; then
#    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0" -O ../data/yelp_review_full_csv.tar.gz && rm -rf /tmp/cookies.txt
#    tar -xvzf ../data/yelp_review_full_csv.tar.gz -C ../data
#    python ../data/data_utils.py
#fi

# Bulk experiments, set right TRAIN/TEST datasets
TRAIN="yelp_review_full.train"
TEST="yelp_review_full.test"
python train.py ../data/$TRAIN ../data/$TEST
python train.py ../data/$TRAIN ../data/$TEST 0.5
python train.py ../data/$TRAIN ../data/$TEST 1.0
python train.py ../data/$TRAIN ../data/$TEST 1.5
python train.py ../data/$TRAIN ../data/$TEST 2.0 

