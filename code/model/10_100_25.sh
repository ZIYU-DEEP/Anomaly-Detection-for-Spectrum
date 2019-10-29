#!/bin/bash
python featurization.py 10 100 25 ryerson normal
python training.py 10 100 25 ryerson 0208_anomaly 25 125 256 200 0
python evaluation.py 10 100 25 ryerson 0208_anomaly 125 256 0
