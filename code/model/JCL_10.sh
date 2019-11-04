#!/bin/bash
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/normal/JCL 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/JCL/sameFBS 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/JCL/diffFBS 10

#python featurization.py 10 100 25 JCL normal
#python featurization.py 10 100 25 JCL/sameFBS abnormal
python training.py 10 100 25 JCL JCL/sameFBS 25 125 256 200 0
python evaluation.py 10 100 25 JCL JCL/sameFBS 125 256 0

#python featurization.py 10 100 25 JCL normal
#python featurization.py 10 100 25 JCL/diffFBS abnormal
python training.py 10 100 25 JCL JCL/diffFBS 25 125 256 200 0
python evaluation.py 10 100 25 JCL JCL/diffFBS 125 256 0

#python featurization.py 10 100 25 JCL normal
#python featurization.py 10 100 25 JCL/diffFBS abnormal
python training.py 10 100 25 JCL JCL/diffFBS 25 125 256 200 0
python evaluation.py 10 100 25 JCL JCL/diffFBS 125 256 0

python featurization.py 10 10000 2500 JCL normal
python featurization.py 10 10000 2500 JCL/sameFBS abnormal
python training.py 10 10000 2500 JCL JCL/sameFBS 2500 12500 256 200 0
python evaluation.py 10 10000 2500 JCL JCL/sameFBS 12500 256 0

python featurization.py 10 10000 2500 JCL normal
python featurization.py 10 10000 2500 JCL/diffFBS abnormal
python training.py 10 10000 2500 JCL JCL/diffFBS 2500 12500 256 200 0
python evaluation.py 10 10000 2500 JCL JCL/diffFBS 12500 256 0