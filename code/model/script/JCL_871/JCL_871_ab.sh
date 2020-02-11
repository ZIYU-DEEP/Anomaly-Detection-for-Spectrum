cd ../../../preprocessing

python downsample_parse_data.py /net/adv_spectrum/data/raw/normal/871_ab 10 6

cd ../model
python featurization.py 10 1000 250 871_ab normal 6