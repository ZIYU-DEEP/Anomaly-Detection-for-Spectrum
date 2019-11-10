##!_bin_bash
python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/normal/JCL_1105C 10

python featurization.py 10 100 25 JCL_1105C normal
python training.py 10 100 25 JCL_1105C 25 125 256 200 0
python evaluation.py 10 100 25 JCL_1105C JCL_sameFBS 125 256 0
python evaluation.py 10 100 25 JCL_1105C JCL_diffFBS 125 256 0

python featurization.py 10 10000 2500 JCL_1105C normal
python training.py 10 10000 2500 JCL_1105C 2500 12500 256 200 0
python evaluation.py 10 10000 2500 JCL_1105C JCL_sameFBS 12500 256 0
python evaluation.py 10 10000 2500 JCL_1105C JCL_diffFBS 12500 256 0