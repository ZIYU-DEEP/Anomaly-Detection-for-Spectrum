#!/bin/bash
python featurization.py 10 100 25 JCL normal
python featurization.py 10 100 25 JCL/sameFBS abnormal
python training.py 10 100 25 JCL JCL/sameFBS 25 125 256 200 0
python evaluation.py 10 100 25 JCL JCL/sameFBS 125 256 0

python featurization.py 10 100 25 JCL normal
python featurization.py 10 100 25 JCL/sameFBS abnormal
python training.py 10 100 25 JCL JCL/sameFBS 25 125 256 200 0
python evaluation.py 10 100 25 JCL JCL/sameFBS 125 256 0