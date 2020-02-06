#!/bin/bash
cd ~/Anomaly-Detection-for-Spectrum/code/model

python featurization.py 10 100 25 ryerson_train normal
python training_joblib.py 10 100 25 ryerson_train 25 125 128 266 1
