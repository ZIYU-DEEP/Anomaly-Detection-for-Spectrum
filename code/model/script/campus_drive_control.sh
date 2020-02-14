cd ~/Anomaly-Detection-for-Spectrum/code/model

python featurization.py 10 1000 250 campus_drive_sigOver_20ms abnormal 4
python featurization.py 10 1000 250 campus_drive_sigOver_10ms abnormal 4
python featurization.py 10 1000 250 campus_drive_sigOver_5ms abnormal 4

python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_sigOver_20ms 1250 128 1
python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_sigOver_10ms 1250 128 1
python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_sigOver_5ms 1250 128 1

python evaluation_joblib_control.py 10 1000 250 campus_drive campus_drive_sigOver_20ms 1250 128 1
python evaluation_joblib_control.py 10 1000 250 campus_drive campus_drive_sigOver_10ms 1250 128 1
python evaluation_joblib_control.py 10 1000 250 campus_drive campus_drive_sigOver_5ms 1250 128 1

cd /net/adv_spectrum/
chmod -R 777 .
