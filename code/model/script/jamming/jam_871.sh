cd ../../../preprocessing
# python jam_ds.py 871_ab 0 5 ask 6
# python jam_ds.py 871_ab 0 5 psk 6
# python jam_ds.py 871_ab 0 5 fsk 6
python jam_ds.py 871_ab 0 5 qam 6

cd ../model
# python featurization.py 10 100 25 871_ab_ask abnormal 6
# python featurization.py 10 1000 250 871_ab_ask abnormal 6
# python featurization.py 10 100 25 871_ab_psk abnormal 6
# python featurization.py 10 1000 250 871_ab_psk abnormal 6
# python featurization.py 10 100 25 871_ab_fsk abnormal 6
# python featurization.py 10 1000 250 871_ab_fsk abnormal 6
python featurization.py 10 100 25 871_ab_qam abnormal 6
python featurization.py 10 1000 250 871_ab_qam abnormal 6

# python jam_ds.py campus_drive 3.4 1.4 ask 6
# python jam_ds.py downtown 3.4 1.4 ask 6
# python jam_ds.py ryerson_ab_train 3.4 1.4 ask 6