cd ../../../preprocessing
cd ../model

python featurization.py 10 1000 250 campus_drive_wn abnormal 6
python featurization.py 10 1000 250 downtown_wn abnormal 6
python featurization.py 10 1000 250 871_ab_wn abnormal 6
python featurization.py 10 1000 250 ryerson_ab_train_wn abnormal 6

python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_wn 1250 128 2
python evaluation_joblib.py 10 1000 250 downtown downtown_wn 1250 128 2
python evaluation_joblib.py 10 1000 250 871 871_ab_wn 1250 128 2
python evaluation_joblib.py 10 1000 250 ryerson ryerson_ab_train_wn 1250 128 2

cd /net/adv_spectrum
chmod 777 -R .