cd ../../preprocessing/

python downsample_parse_data.py /net/adv_spectrum/data/raw/normal/ryerson_train 10
cd ../model
python featurization.py 10 1000 250 ryerson_train normal
cd ../preprocessing
python downsample_parse_data.py /net/adv_spectrum/data/raw/normal/ryerson_test 10
cd ../model
python featurization.py 10 1000 250 ryerson_test normal
cd ../preprocessing
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_ryerson_t1 10
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_ryerson_t2 10
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_JCL 10
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_downtown 10
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_test_searle 10

cd ../model
python featurization.py 10 1000 250 ryerson_test_ryerson_t1 abnormal
python featurization.py 10 1000 250 ryerson_test_ryerson_t2 abnormal
python featurization.py 10 1000 250 ryerson_test_JCL abnormal
python featurization.py 10 1000 250 ryerson_test_downtown abnormal
python featurization.py 10 1000 250 ryerson_test_searle abnormal