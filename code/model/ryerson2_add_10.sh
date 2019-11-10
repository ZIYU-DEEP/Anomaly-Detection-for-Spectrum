##!_bin_bash
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_same/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G-3/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G0/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G3/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G6/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G9/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G12/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G15/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G18/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G21/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson2_diff_G24/ 10

python featurization.py 10 100 25 ryerson2_same abnormal
python featurization.py 10 100 25 ryerson2_diff_G-3 abnormal
python featurization.py 10 100 25 ryerson2_diff_G0 abnormal
python featurization.py 10 100 25 ryerson2_diff_G3 abnormal
python featurization.py 10 100 25 ryerson2_diff_G6 abnormal
python featurization.py 10 100 25 ryerson2_diff_G9 abnormal
python featurization.py 10 100 25 ryerson2_diff_G12 abnormal
python featurization.py 10 100 25 ryerson2_diff_G15 abnormal
python featurization.py 10 100 25 ryerson2_diff_G18 abnormal
python featurization.py 10 100 25 ryerson2_diff_G21 abnormal
python featurization.py 10 100 25 ryerson2_diff_G24 abnormal

python featurization.py 10 1000 250 ryerson2_same abnormal
python featurization.py 10 1000 250 ryerson2_diff_G-3 abnormal
python featurization.py 10 1000 250 ryerson2_diff_G0 abnormal
python featurization.py 10 1000 250 ryerson2_diff_G3 abnormal
python featurization.py 10 1000 250 ryerson2_diff_G6 abnormal
python featurization.py 10 1000 250 ryerson2_diff_G9 abnormal
python featurization.py 10 1000 250 ryerson2_diff_G12 abnormal
python featurization.py 10 1000 250 ryerson2_diff_G15 abnormal
python featurization.py 10 1000 250 ryerson2_diff_G18 abnormal
python featurization.py 10 1000 250 ryerson2_diff_G21 abnormal
python featurization.py 10 1000 250 ryerson2_diff_G24 abnormal
