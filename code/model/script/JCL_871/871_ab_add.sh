cd ../../../preprocessing
a="/net/adv_spectrum/data/raw/abnormal/871_ab_"
b="871_ab_"
a+=$1
echo $a
python add_raw.py 6 $1 12 5
python downsample_parse_data.py $a 10 5
rm -rf $a

cd ../model
b+=$1
echo $b
python featurization.py 10 1000 250 $b abnormal 5
chmod 777 -R /net/adv_spectrum