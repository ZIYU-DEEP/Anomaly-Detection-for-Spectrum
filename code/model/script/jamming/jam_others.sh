cd ../../../preprocessing
# python jam_ds.py $1 3.4 1.4 ask 6
# python jam_ds.py $1 3.4 1.4 psk 6
# python jam_ds.py $1 3.4 1.4 fsk 6
python jam_ds.py $1 3.4 1.4 qam 6

cd ../model
# python featurization.py 10 100 25 "$1_ask" abnormal 6
# python featurization.py 10 1000 250 "$1_ask" abnormal 6
# python featurization.py 10 100 25 "$1_psk" abnormal 6
# python featurization.py 10 1000 250 "$1_psk" abnormal 6
# python featurization.py 10 100 25 "$1_fsk" abnormal 6
# python featurization.py 10 1000 250 "$1_fsk" abnormal 6
python featurization.py 10 100 25 "$1_qam" abnormal 6
python featurization.py 10 1000 250 "$1_qam" abnormal 6