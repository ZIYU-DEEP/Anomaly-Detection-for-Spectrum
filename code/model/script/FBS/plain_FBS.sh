cd ../../../preprocessing
a="/net/adv_spectrum/data/raw/abnormal/"
a+=$1
echo $a
python downsample_parse_data.py $a 10 5

cd ../model
python featurization.py 10 1000 250 $1 abnormal 5