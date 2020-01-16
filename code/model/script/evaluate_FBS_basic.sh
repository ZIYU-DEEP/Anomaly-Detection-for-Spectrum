cd ..
python featurization.py 10 1000 250 FBS_basic abnormal 4
python evaluation_joblib.py 10 1000 250 ryerson_train FBS_basic 250 128 1