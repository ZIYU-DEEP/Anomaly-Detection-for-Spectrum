#!/bin/bash
cd ../../preprocessing/

python downsample_parse_data.py /net/adv_spectrum/data/raw/normal/ryerson 100
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/0208_anomaly 100

python downsample_parse_data.py /net/adv_spectrum/data/raw/normal/ryerson 500
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/0208_anomaly 500

python downsample_parse_data.py /net/adv_spectrum/data/raw/normal/ryerson 1000
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/0208_anomaly 1000

python downsample_parse_data.py /net/adv_spectrum/data/raw/normal/ryerson 1500
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/0208_anomaly 1500

python downsample_parse_data.py /net/adv_spectrum/data/raw/normal/ryerson 2000
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/0208_anomaly 2000
