cd ~/Anomaly-Detection-for-Spectrum/code/preprocessing
a="/net/adv_spectrum/data/raw/abnormal/"
b="$1_"
echo $b
a+=$1
a+="_"
a+=$2
echo $a
python add_raw.py 6 $1 $2 6 5
python downsample_parse_data.py $a 10 6
rm -rf $a

cd ../model
b+=$2
echo $b
python featurization.py 10 1000 250 $b abnormal 6
chmod 777 -R /net/adv_spectrum