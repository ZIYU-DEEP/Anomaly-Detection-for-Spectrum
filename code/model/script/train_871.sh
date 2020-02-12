cd ..

python featurization.py 10 100 25 871 normal 5
python training.py 10 100 25 871 25 125 256 300 3
python training.py 10 1000 250 871 250 1250 256 300 3
