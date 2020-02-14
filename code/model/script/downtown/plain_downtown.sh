cd ../../../preprocessing
a="/net/adv_spectrum/data/raw/normal/downtown"
python downsample_parse_data.py $a 10 5

cd ../model
python featurization.py 10 1000 250 downtown normal 5
chmod 777 -R /net/adv_spectrum
python featurization.py 10 100 25 downtown normal 5
chmod 777 -R /net/adv_spectrum