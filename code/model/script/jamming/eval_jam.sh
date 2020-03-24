cd ../../../preprocessing
cd ../model

# python evaluation_joblib.py 10 1000 250 $1 "$2_wn" 250 128 $3
# python evaluation_joblib.py 10 1000 250 $1 "$2_ask" 250 128 $3
# python evaluation_joblib.py 10 1000 250 $1 "$2_psk" 250 128 $3
# python evaluation_joblib.py 10 1000 250 $1 "$2_fsk" 250 128 $3
python evaluation_joblib.py 10 1000 250 ryerson_train "ryerson_ab_train_$1" 250 128 $2
python evaluation_joblib.py 10 1000 250 871 "871_ab_$1" 250 128 $2
python evaluation_joblib.py 10 1000 250 downtown "downtown_$1" 250 128 $2
python evaluation_joblib.py 10 1000 250 campus_drive "campus_drive_$1" 250 128 $2

chmod 777 -R /net/adv_spectrum

# python evaluation_joblib.py 10 100 25 $1 "$2_wn" 25 128 $3
# python evaluation_joblib.py 10 100 25 $1 "$2_ask" 25 128 $3
# python evaluation_joblib.py 10 100 25 $1 "$2_psk" 25 128 $3
# python evaluation_joblib.py 10 100 25 $1 "$2_fsk" 25 128 $3
# python evaluation_joblib.py 10 100 25 $1 "$2_qam" 25 128 $3

# chmod 777 -R /net/adv_spectrum