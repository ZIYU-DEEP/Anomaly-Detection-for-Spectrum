cd ../../../preprocessing

cd ../model
python featurization.py 10 1000 250 871_ab_sigOver_5ms abnormal 6
python evaluation_joblib.py 10 1000 250 871 871_ab_sigOver_5ms 250 128 1

cd /net/adv_spectrum
chmod 777 -R .