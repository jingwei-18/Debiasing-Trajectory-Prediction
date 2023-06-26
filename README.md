# Debiasing Trajectory Prediction via Disentangling Demographic Properties for Equality

## Project details

**Project of FIB-LAB, Tsinghua University** 

**Developer names: ZhiHong Lu, JingWei Wang, JiaHui Gong**

**Advisor: Prof. Yong Li**

## Implementation

**Source Code: https://github.com/jingwei-18/Debiasing-Trajectory-Prediction**

**Paper:**

**P.S. Because the datasets used (Mobile Operator Dataset and Social Network Platform Dataset) are not available to pulic, an small example of dataset format is provided (See documentation of user2info.json and user2traj.json)**

## Documentation

- **model.py** : Python file that includes defintion for Seq2Seq, Seq2Seq with property estimation, Seq2Seq with adversarial loss models
- **train_no.py**: Python script that trains the Seq2Seq model
- **train_distan.py**: Python script that trains the Seq2Seq2 with estimation and Seq2Seq with adversarial loss model
- **test_no.py**: Python script that generates test results for Seq2Seq model
- **test_no.py**: Python script that genderates test results for Seq2Seq2 with estimation and Seq2Seq with adversarial loss model
- **result_*.csv**: csv files that stores the experimental results listed in the paper:
    - With '_no' means Seq2Seq model
    - With '_dis' means Seq2Seq with disentangling adversarial loss model
    - Without '_no' and '_dis' means Seq2Seq2 with estimation
    - With '_Ten' means data of SNP dataset
    - without '_Ten' means data of MO dataset
    - **requirements.txt**: The reliables for this project to work