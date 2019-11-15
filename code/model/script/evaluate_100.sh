#!/bin/bash
cd ../../preprocessing/

python downsample_parse_data.py /net/adv_spectrum/data/raw/normal/ryerson 100
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/0208_anomaly 100
