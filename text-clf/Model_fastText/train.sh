#!/bin/bash

# Bulk experiments, set right TRAIN/TEST datasets
TRAIN="yelp_review_full.train"
TEST="yelp_review_full.test"
python train.py ../data/$TRAIN ../data/$TEST
python train.py ../data/$TRAIN ../data/$TEST 0.5
python train.py ../data/$TRAIN ../data/$TEST 1.0
python train.py ../data/$TRAIN ../data/$TEST 1.5
python train.py ../data/$TRAIN ../data/$TEST 2.0

