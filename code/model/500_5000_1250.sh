#!/bin/bash
python featurization.py 500 5000 1250 ryerson normal
python training.py 500 5000 1250 ryerson 0208_anomaly 1250 6250 256 200 1
python evaluation.py 500 5000 1250 ryerson 0208_anomaly 6250 256 1
