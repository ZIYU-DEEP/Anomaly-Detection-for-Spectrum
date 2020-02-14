cd  ../../

# python featurization.py 10 100 25 downtown_LOS-5M-USRP1 abnormal 6
# python featurization.py 10 100 25 downtown_LOS-5M-USRP2 abnormal 6
# python featurization.py 10 100 25 downtown_LOS-5M-USRP3 abnormal 6
# python featurization.py 10 100 25 downtown_NLOS-5M-USRP1 abnormal 6
# python featurization.py 10 100 25 downtown_Dynamics-5M-USRP1 abnormal 6

python training_joblib.py 10 100 25 downtown 25 125 256 300 0
python training_joblib.py 10 1000 250 downtown 250 1250 256 300 0

python evaluation_joblib.py 10 100 25 downtown downtown_LOS-5M-USRP1 125 128 0
python evaluation_joblib.py 10 100 25 downtown downtown_LOS-5M-USRP2 125 128 0
python evaluation_joblib.py 10 100 25 downtown downtown_LOS-5M-USRP3 125 128 0
python evaluation_joblib.py 10 100 25 downtown downtown_NLOS-5M-USRP1 125 128 0
python evaluation_joblib.py 10 100 25 downtown downtown_Dynamics-5M-USRP1 125 128 0

python evaluation_joblib.py 10 1000 250 downtown downtown_LOS-5M-USRP1 1250 128 0
python evaluation_joblib.py 10 1000 250 downtown downtown_LOS-5M-USRP2 1250 128 0
python evaluation_joblib.py 10 1000 250 downtown downtown_LOS-5M-USRP3 1250 128 0
python evaluation_joblib.py 10 1000 250 downtown downtown_NLOS-5M-USRP1 1250 128 0
python evaluation_joblib.py 10 1000 250 downtown downtown_Dynamics-5M-USRP1 1250 128 0

cd /net/adv_spectrum/
chmod -R 777 .
