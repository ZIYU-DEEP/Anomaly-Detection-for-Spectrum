#!/bin/bash
cd ..
python evaluation_joblib.py 10 1000 250 ryerson_train ryerson_test_downtown 250 128 3
python evaluation_joblib.py 10 1000 250 ryerson_train ryerson_test_JCL 250 128 3
python evaluation_joblib.py 10 1000 250 ryerson_train ryerson_test_searle 250 128 3
