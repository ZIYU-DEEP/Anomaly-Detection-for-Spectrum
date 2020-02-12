cd ..

python training_joblib_control.py 10 100 25 ryerson_train 25 125 128 300 3
python evaluation_joblib_control.py 10 100 25 ryerson_train ryerson_ab_train_sigOver 125 128 3 
