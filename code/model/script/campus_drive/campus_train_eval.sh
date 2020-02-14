cd ../../
python training_joblib.py 10 100 25 campus_drive 25 125 256 300 2
python training_joblib.py 10 1000 250 campus_drive 250 1250 256 300 2

python evaluation_joblib.py 10 100 25 campus_drive campus_drive_LOS-5M-USRP1 125 128 2
python evaluation_joblib.py 10 100 25 campus_drive campus_drive_LOS-5M-USRP2 125 128 2
python evaluation_joblib.py 10 100 25 campus_drive campus_drive_LOS-5M-USRP3 125 128 2
python evaluation_joblib.py 10 100 25 campus_drive campus_drive_NLOS-5M-USRP1 125 128 2
python evaluation_joblib.py 10 100 25 campus_drive campus_drive_Dynamics-5M-USRP1 125 128 2

python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_LOS-5M-USRP1 1250 128 2
python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_LOS-5M-USRP2 1250 128 2
python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_LOS-5M-USRP3 1250 128 2
python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_NLOS-5M-USRP1 1250 128 2
python evaluation_joblib.py 10 1000 250 campus_drive campus_drive_Dynamics-5M-USRP1 1250 128 2

cd /net/adv_spectrum/
chmod -R 777 .
