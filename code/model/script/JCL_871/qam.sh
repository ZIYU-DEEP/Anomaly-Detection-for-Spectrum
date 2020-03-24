cd ../../../preprocessing
python jam_ds.py 871_ab 0 5 qam 6
python featurization.py 10 100 25 871_ab_qam abnormal 6
python featurization.py 10 1000 250 871_ab_qam abnormal 6