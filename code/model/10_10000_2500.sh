#!/bin/bash
python featurization.py 10 10000 2500 ryerson normal
python training.py 10 10000 2500 ryerson 0208_anomaly 2500 12500 256 200 2
python evaluation.py 10 10000 2500 ryerson 0208_anomaly 12500 256 2
