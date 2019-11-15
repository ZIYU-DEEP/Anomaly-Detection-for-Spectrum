##!_bin_bash
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/normal/JCL 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/JCL/sameFBS 10
#python ../preprocessing/downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/JCL/diffFBS 10
cd ..

python featurization.py 10 100 25 JCL normal
python featurization.py 10 100 25 JCL_sameFBS abnormal
python featurization.py 10 100 25 JCL_diffFBS abnormal
python training.py 10 100 25 JCL 25 125 256 200 0
python evaluation.py 10 100 25 JCL JCL_sameFBS 125 256 0
python evaluation.py 10 100 25 JCL JCL_diffFBS 125 256 0

python featurization.py 10 10000 2500 JCL normal
python featurization.py 10 10000 2500 JCL_sameFBS abnormal
python featurization.py 10 10000 2500 JCL_diffFBS abnormal
python training.py 10 10000 2500 JCL 2500 12500 256 200 0
python evaluation.py 10 10000 2500 JCL JCL_sameFBS 12500 256 0
python evaluation.py 10 10000 2500 JCL JCL_diffFBS 12500 256 0