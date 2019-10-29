#!/bin/bash
python featurization.py 1000 10000 2500 ryerson normal
python training.py 1000 10000 2500 ryerson 0208_anomaly 2500 12500 256 200 2
python evaluation.py 1000 10000 2500 ryerson 0208_anomaly 12500 256 2
