cd ../../../preprocessing

cd ../model
python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_sigOver_10ms 250 128 2
python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_sigOver_5ms 250 128 2
python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_sigOver_20ms 250 128 2
python evaluation_joblib.py 10 1000 250 downtown downtown_sigOver_10ms 250 128 2
python evaluation_joblib.py 10 1000 250 downtown downtown_sigOver_5ms 250 128 2
python evaluation_joblib.py 10 1000 250 downtown downtown_sigOver_20ms 250 128 2

cd /net/adv_spectrum
chmod 777 -R .