cd ../../preprocessing

python add_raw.py 6 JCL 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_JCL 10 6

cd ../model
python featurization.py 10 1000 250 ryerson_test_JCL abnormal 6
python evaluation_joblib.py 10 1000 250 ryerson_train ryerson_test_JCL 250 128 2