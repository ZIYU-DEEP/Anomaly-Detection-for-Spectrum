# Progress Report - Nov 10, 2019

## 0. General Settings

- **Training Data**: 72 samples from 72-hour Ryerson data 
  - Select a file at each hour, each file contains 40s' data
  - The trained model can be approximately conceived as a general model robust on time

- **BS Data**: A consecutive sequence of files from the 72-hour Ryerson data
- **FBS Data**: Another consecutive sequence of files from the 72-hour Ryerson data



## 1. Plot Analysis

### 1.1. Time-MSE Plot of `BS + Periodical FBS` Example

<img src="/Users/ziyuye/Library/Application Support/typora-user-images/image-20191110235211361.png" alt="image-20191110235211361" style="zoom:150%;" />

> `downsample_ratio` = 10,  `window_size` = 1000, `shift_size` = 250
>
> `BS_file` = 1518560024_880M_5m, `FBS_file` = 1518568777_880M_5m , `power_gain` = 24
>
> The two files has a time interval of 2.5 hrs. Results with other power gains will be provided later.
>
> Let `jump_start` denotes the point that the attacker starts transmitting, and `jump_end` denotes the point that the attacker ends transmitting. 

**Findings**:

- At the jump points (when FBS is added), there is a sudden **surge of MSE values**.
  - As sliding window is used, so all the MSEs nearby the jump point will be likely to get sudden surge.
  - We may not be able to distinguish `jump_start` and `jump_end` simply by MSE values. This may raise some small problems, as we may fail to identity which subsequence is normal and which subsequence is abnormal directly from MSE values.
- The MSE of all anomaly **(`BS + FBS`)** **cannot be distinguished** from the MSE of all normal **(`BS`)**.
  - This means the feature representation of **`BS`** and **`BS + FBS`** is similar.
    - A possible cause may be that **`FBS`** is simply a replay of normal data.
    - When **`FBS`** is another type of signal, the superposition signal **`BS + FBS`** may be in a different feature space of **`BS`**. Experiments needed.

### 1.2. MSE CDF Plot Example

In progress.



## 2. What We Could Do Next

Firstly, we'll test the Ryerson BS + FBS data with different power gains, to see how that affects the detection results / MSE behaviors. And we'll also check the MSE CDF. (Already in progress. Waiting for babygroot's results.)

Secondly, we may change the FBS data to other types of signals (e.g. there are some other popular types of FBS), and see how if the new **`BS + FBS`** can be distinguised from **`BS`**.

- If still not, some potential directions:
  - Loss function design: MSE may not be our best choice as the loss. Possible to design the loss according to the mathematical or geometric properties of signal superposition.
  - Network achitecture design: autoencoder-based method may help distinguish features better.

On the other hand, suppose attackers can inject signals by three different patterns:

- **Pattern 1**: Injecting in a jump-point manner
- **Pattern 2**: Gradually injecting, no jump points, but has trends
- **Pattern 3**: Unsteadily injecting, no jump points nor trends, but non-stationary

We can try to inject **`FBS`** by pattern 2 and pattern 3 and see how our detector works.