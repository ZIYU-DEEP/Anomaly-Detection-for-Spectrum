cd ..
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_ts1/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_ts3/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_ts5/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_ts7/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_ts9/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_ts11/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_ts13/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_ts15/ 10

python featurization.py 10 1000 250 ryerson_test_ts1 abnormal
python featurization.py 10 1000 250 ryerson_test_ts3 abnormal
python featurization.py 10 1000 250 ryerson_test_ts5 abnormal
python featurization.py 10 1000 250 ryerson_test_ts7 abnormal
python featurization.py 10 1000 250 ryerson_test_ts9 abnormal
python featurization.py 10 1000 250 ryerson_test_ts11 abnormal
python featurization.py 10 1000 250 ryerson_test_ts13 abnormal
python featurization.py 10 1000 250 ryerson_test_ts15 abnormal