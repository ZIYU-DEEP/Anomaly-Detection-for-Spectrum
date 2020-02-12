cd ..
python training.py 10 100 25 downtown 25 125 256 300 3
python training.py 10 1000 250 downtown 250 1250 256 300 3

python evaluation_joblib.py 10 100 25 downtown downtown_ab_LOS-5M-USRP1 125 128 3
python evaluation_joblib.py 10 100 25 downtown downtown_ab_LOS-5M-USRP2 125 128 3
python evaluation_joblib.py 10 100 25 downtown downtown_ab_LOS-5M-USRP3 125 128 3
python evaluation_joblib.py 10 100 25 downtown downtown_ab_NLOS-5M-USRP1 125 128 3
python evaluation_joblib.py 10 100 25 downtown downtown_ab_Dynamics-5M-USRP1 125 128 3

python evaluation_joblib.py 10 1000 250 downtown downtown_ab_LOS-5M-USRP1 1250 128 3
python evaluation_joblib.py 10 1000 250 downtown downtown_ab_LOS-5M-USRP2 1250 128 3
python evaluation_joblib.py 10 1000 250 downtown downtown_ab_LOS-5M-USRP3 1250 128 3
python evaluation_joblib.py 10 1000 250 downtown downtown_ab_NLOS-5M-USRP1 1250 128 3
python evaluation_joblib.py 10 1000 250 downtown downtown_ab_Dynamics-5M-USRP1 1250 128 3

cd /net/adv_spectrum/
chmod -R 777 .
