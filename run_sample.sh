#!/usr/bin/env bash
echo "Step 1: make unwraps..."
mkdir -p data/sample/unwraps
python3 step1_make_unwraps.py data/sample/frame_data_.pkl data/sample/frames data/sample/segmentations data/sample/unwraps
echo "Step 2: make priors..."
python3 step2_segm_vote_gmm.py data/sample/unwraps data/sample/segm.png data/sample/gmm.pkl
echo "Step 3: stitch texture..."
python3 step3_stitch_texture.py data/sample/unwraps data/sample/segm.png data/sample/gmm.pkl data/sample/sample_texture.jpg
