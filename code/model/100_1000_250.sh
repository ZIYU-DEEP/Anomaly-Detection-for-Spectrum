#!/bin/bash
python featurization.py 100 1000 250 ryerson normal
python training.py 100 1000 250 ryerson 0208_anomaly 250 1250 256 200 0
python evaluation.py 100 1000 250 ryerson 0208_anomaly 1250 256 0
