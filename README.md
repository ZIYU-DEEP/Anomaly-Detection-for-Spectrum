# Anomaly-Detection-for-Spectrum
## 1. Introduction

This is repository for anomaly detection for spectrum data. Currently, the model part has four components:

0. [utils.py](https://github.com/ZIYU-DEEP/Anomaly-Detection-for-Spectrum/blob/master/code/model/utils.py)
1. [featurization.py](https://github.com/ZIYU-DEEP/Anomaly-Detection-for-Spectrum/blob/master/code/model/featurization.py)
2. [training.py](https://github.com/ZIYU-DEEP/Anomaly-Detection-for-Spectrum/blob/master/code/model/training.py)
3. [evaluation.py](https://github.com/ZIYU-DEEP/Anomaly-Detection-for-Spectrum/blob/master/code/model/evaluation.py)





## 2. Model

### 2.1. Featurization

> **Code**: [featurization.py](https://github.com/ZIYU-DEEP/Anomaly-Detection-for-Spectrum/blob/master/code/model/featurization.py)
>
> **Input**: Downsampled normal and abnormal data 
>
> **Output**: Featurized normal and abnormal data

This step is to featurize and normalize the downsampled data. The output to save is in the format of list of numpy arrays.

In the directory of the code script, run the following code in terminal:

```
$ python featurization.py downsample_ratio window_size predict_size folder data_type
```

A sample input parameter would be:

- `downsample_ratio` = 10
- `window_size` = 100
- `predict_size` = 25
- `folder` = ryerson (or 0208_anomaly)
- `data_type` = normal (or abnormal)

Thus results in:

```
$ python featurization.py 10 100 25 ryerson normal
```



### 2.2. Training

> **Code**: [training.py](https://github.com/ZIYU-DEEP/Anomaly-Detection-for-Spectrum/blob/master/code/model/training.py)
>
> **Input**: Featurized normal data (and abnormal data - if evaluation needed)
>
> **Output**: Trained model, and `full_x_valid` (a numpy array for future threshold use)

This step is to fit, validate and evaluate the model. In addition, it would also produce a array of data to validate the model. When validating the model on the array, we will  choose a certain false positive rate and set the corresponding error as the anomaly detection threshold.

In the directory of the code script, run the following code in terminal:

```
$ python training.py downsample_ratio window_size predict_size normal_folder shift_train shift_eval batch_size epochs gpu_no
```

A sample input parameter would be:

- `downsample_ratio` = 10
- `window_size` = 100
- `predict_size` = 25
- `normal_folder` = ryerson
- `shift_train` = 25
- `shift_eval` = 125
- `batch_size` = 256
- `epochs` = 50
- `gpu_no` = 3

Thus results in:

```
$ python training.py 10 100 25 ryerson 25 125 256 50 3
```

**Note**: We suggest to use sliding window to train, such that you might specify the value of  `shift_train` equal to `predict_size`; and use non-sliding window to evaluate, such that you might specify the value of `shift_eval` equal to `window_size + predict_size`.

Be sure that the input parameter in this step is consistent with the previous step.



### 2.3. Evaluation

> **Code**: [evaluation.py](https://github.com/ZIYU-DEEP/Anomaly-Detection-for-Spectrum/blob/master/code/model/evaluation.py)
>
> **Input**: Model, the list of abnormal series, and the array of `full_x_valid`
>
> **Output**: The DataFrame of valid errors, the DataFrames of anomaly errors and the corresponding CDF plot

This step is to evaluate the model's anomaly detection performance on different anomaly inputs.

In the directory of the code script, run the following code in terminal:

```
$ python evaluation.py downsample_ratio window_size predict_size normal_folder anomaly_folder shift_eval batch_size gpu_no
```

A sample input parameter would be:

- `downsample_ratio` = 10
- `window_size` = 100
- `predict_size` = 25
- `normal_folder` = ryerson
- `anomaly_folder` = 0208_anomaly
- `shift_eval` = 125
- `batch_size` = 256
- `gpu_no` = 3

Thus results in:

```
$ python evaluation.py 10 100 25 ryerson 0208_anomaly 125 256 3
```

Be sure that the input parameter in this step is consistent with the previous step.