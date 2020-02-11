cd ../../../preprocessing

python downsample_parse_data.py /net/adv_spectrum/data/raw/normal/871 10 6

cd ../model
python featurization.py 10 1000 250 871 normal 6