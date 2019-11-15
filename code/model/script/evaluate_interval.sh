#!/bin/bash
cd ..

python evaluation_interval.py 10 1000 250 ryerson_all ryerson2_diff_G0 250 128 3
python evaluation_interval.py 10 1000 250 ryerson_all ryerson2_diff_G3 250 128 3
python evaluation_interval.py 10 1000 250 ryerson_all ryerson2_diff_G6 250 128 3