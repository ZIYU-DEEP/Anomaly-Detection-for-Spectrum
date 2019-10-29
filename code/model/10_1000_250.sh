#!/bin/bash
python featurization.py 10 1000 250 ryerson normal
python training.py 10 1000 250 ryerson 0208_anomaly 250 1250 256 200 0
python evaluation.py 10 1000 250 ryerson 0208_anomaly 1250 256 0
