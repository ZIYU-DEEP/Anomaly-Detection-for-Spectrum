cd ../../../preprocessing

# python relay_raw.py 6 21
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_21dB 10 6

cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_21dB abnormal 6