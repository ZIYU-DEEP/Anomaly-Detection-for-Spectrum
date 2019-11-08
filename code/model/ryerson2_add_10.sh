##!_bin_bash
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_same/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G-3/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G0/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G3/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G6/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G9/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G12/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G15/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G18/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G21/ 10
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G24/ 10

python featurization.py 10 100 25 ryerson2_same normal
python featurization.py 10 100 25 ryerson2_diff_G-3 normal
python featurization.py 10 100 25 ryerson2_diff_G0 normal
python featurization.py 10 100 25 ryerson2_diff_G3 normal
python featurization.py 10 100 25 ryerson2_diff_G6 normal
python featurization.py 10 100 25 ryerson2_diff_G9 normal
python featurization.py 10 100 25 ryerson2_diff_G12 normal
python featurization.py 10 100 25 ryerson2_diff_G15 normal
python featurization.py 10 100 25 ryerson2_diff_G18 normal
python featurization.py 10 100 25 ryerson2_diff_G21 normal
python featurization.py 10 100 25 ryerson2_diff_G24 normal