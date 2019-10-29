#!/bin/bash
# The compilation of downsample_ratio 10

# 100 25
#!/bin/bash
python featurization.py 10 100 25 ryerson normal
python training.py 10 100 25 ryerson 0208_anomaly 25 125 256 200 0
python evaluation.py 10 100 25 ryerson 0208_anomaly 125 256 0

# 1000 250
#!/bin/bash
python featurization.py 10 1000 250 ryerson normal
python training.py 10 1000 250 ryerson 0208_anomaly 250 1250 256 200 0
python evaluation.py 10 1000 250 ryerson 0208_anomaly 1250 256 0

# 5000 1250
#!/bin/bash
python featurization.py 10 5000 1250 ryerson normal
python training.py 10 5000 1250 ryerson 0208_anomaly 1250 6250 256 200 1
python evaluation.py 10 5000 1250 ryerson 0208_anomaly 6250 256 1

# 10000 2500
#!/bin/bash
python featurization.py 10 10000 2500 ryerson normal
python training.py 10 10000 2500 ryerson 0208_anomaly 2500 12500 256 200 2
python evaluation.py 10 10000 2500 ryerson 0208_anomaly 12500 256 2

# 20000 5000
#!/bin/bash
python featurization.py 10 20000 5000 ryerson normal
python training.py 10 20000 5000 ryerson 0208_anomaly 5000 25000 256 200 3
python evaluation.py 10 20000 5000 ryerson 0208_anomaly 25000 256 3
