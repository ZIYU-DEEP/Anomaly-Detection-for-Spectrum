##!_bin_bash
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix_G-3/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix_G0/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix_G3/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix_G6/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix_G9/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix_G12/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix_G15/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix_G18/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix_G21/ 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix_G24/ 10

python featurization.py 10 100 25 ry2_jcl_mix_G-3 abnormal
python featurization.py 10 100 25 ry2_jcl_mix_G0 abnormal
python featurization.py 10 100 25 ry2_jcl_mix_G3 abnormal
python featurization.py 10 100 25 ry2_jcl_mix_G6 abnormal
python featurization.py 10 100 25 ry2_jcl_mix_G9 abnormal
python featurization.py 10 100 25 ry2_jcl_mix_G12 abnormal
python featurization.py 10 100 25 ry2_jcl_mix_G15 abnormal
python featurization.py 10 100 25 ry2_jcl_mix_G18 abnormal
python featurization.py 10 100 25 ry2_jcl_mix_G21 abnormal
python featurization.py 10 100 25 rry2_jcl_mix_G24 abnormal

python evaluation_interval.py 10 100 25 ryerson_all ry2_jcl_mix_G-3 25 128 3
python evaluation_interval.py 10 100 25 ryerson_all ry2_jcl_mix_G0 25 128 3
python evaluation_interval.py 10 100 25 ryerson_all ry2_jcl_mix_G3 25 128 3
python evaluation_interval.py 10 100 25 ryerson_all ry2_jcl_mix_G6 25 128 3
python evaluation_interval.py 10 100 25 ryerson_all ry2_jcl_mix_G9 25 128 3
python evaluation_interval.py 10 100 25 ryerson_all ry2_jcl_mix_G12 25 128 3
python evaluation_interval.py 10 100 25 ryerson_all ry2_jcl_mix_G15 25 128 3
python evaluation_interval.py 10 100 25 ryerson_all ry2_jcl_mix_G18 25 128 3
python evaluation_interval.py 10 100 25 ryerson_all ry2_jcl_mix_G21 25 128 3
python evaluation_interval.py 10 100 25 ryerson_all ry2_jcl_mix_G24 25 128 3

python featurization.py 10 1000 250 ry2_jcl_mix_G-3 abnormal
python featurization.py 10 1000 250 ry2_jcl_mix_G0 abnormal
python featurization.py 10 1000 250 ry2_jcl_mix_G3 abnormal
python featurization.py 10 1000 250 ry2_jcl_mix_G6 abnormal
python featurization.py 10 1000 250 ry2_jcl_mix_G9 abnormal
python featurization.py 10 1000 250 ry2_jcl_mix_G12 abnormal
python featurization.py 10 1000 250 ry2_jcl_mix_G15 abnormal
python featurization.py 10 1000 250 ry2_jcl_mix_G18 abnormal
python featurization.py 10 1000 250 ry2_jcl_mix_G21 abnormal
python featurization.py 10 1000 250 rry2_jcl_mix_G24 abnormal

python evaluation_interval.py 10 1000 250 ryerson_all ry2_jcl_mix_G-3 250 128 3
python evaluation_interval.py 10 1000 250 ryerson_all ry2_jcl_mix_G0 250 128 3
python evaluation_interval.py 10 1000 250 ryerson_all ry2_jcl_mix_G3 250 128 3
python evaluation_interval.py 10 1000 250 ryerson_all ry2_jcl_mix_G6 250 128 3
python evaluation_interval.py 10 1000 250 ryerson_all ry2_jcl_mix_G9 250 128 3
python evaluation_interval.py 10 1000 250 ryerson_all ry2_jcl_mix_G12 250 128 3
python evaluation_interval.py 10 1000 250 ryerson_all ry2_jcl_mix_G15 250 128 3
python evaluation_interval.py 10 1000 250 ryerson_all ry2_jcl_mix_G18 250 128 3
python evaluation_interval.py 10 1000 250 ryerson_all ry2_jcl_mix_G21 250 128 3
python evaluation_interval.py 10 1000 250 ryerson_all ry2_jcl_mix_G24 250 128 3