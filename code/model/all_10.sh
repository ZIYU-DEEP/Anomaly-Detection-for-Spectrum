#!/bin/bash

python featurization.py 100 1000 250 ryerson normal
python featurization.py 100 1000 250 0208_anomaly abnormal
python training.py 100 1000 250 ryerson 0208_anomaly 250 1250 256 200 0
python evaluation.py 100 1000 250 ryerson 0208_anomaly 1250 256 0

python featurization.py 500 5000 1250 ryerson normal
python featurization.py 500 5000 1250 0208_anomaly abnormal
python training.py 500 5000 1250 ryerson 0208_anomaly 1250 6250 256 200 0
python evaluation.py 500 5000 1250 ryerson 0208_anomaly 6250 256 0

python featurization.py 1000 10000 2500 ryerson normal
python featurization.py 1000 10000 2500 0208_anomaly abnormal
python training.py 1000 10000 2500 ryerson 0208_anomaly 2500 12500 256 200 0
python evaluation.py 1000 10000 2500 ryerson 0208_anomaly 12500 256 0

python featurization.py 2000 20000 5000 ryerson normal
python featurization.py 2000 20000 5000 0208_anomaly abnormal
python training.py 2000 20000 5000 ryerson 0208_anomaly 5000 25000 256 200 0
python evaluation.py 2000 20000 5000 ryerson 0208_anomaly 25000 256 0
