#!/bin/bash
python featurization.py 2000 20000 5000 ryerson normal
python training.py 2000 20000 5000 ryerson 0208_anomaly 5000 25000 256 200 3
python evaluation.py 2000 20000 5000 ryerson 0208_anomaly 25000 256 3
