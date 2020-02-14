cd ../../../preprocessing
a="/net/adv_spectrum/data/raw/normal/campus_drive"
python downsample_parse_data.py $a 10 6

cd ../model
python featurization.py 10 1000 250 campus_drive normal 6
chmod 777 -R /net/adv_spectrum
python featurization.py 10 100 25 campus_drive normal 6
chmod 777 -R /net/adv_spectrum