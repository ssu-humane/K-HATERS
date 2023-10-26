# K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings

This repository provides the code and dataset for our EMNLP'23 findings paper.

## Data (K-HATERS)
### How to use dataset throught Hugging Face datasets library
```python
from datasets import load_dataset

data = load_dataset('humane-lab/K-HATERS')
```
```python
>>> data['train'][0]
{'token_ids': [2, 9402, 8506, 2240, 4007, 1966, 28302, 17604, 4673, 8630, 14694, 1410, 4322, 3142, 4767, 1505, 2539, 8027, 2988, 24907, 2656, 4327, 4152, 3438, 4123, 1465, 2584, 802, 2195, 1862, 4019, 17582, 774, 4038, 4038, 1304, 4010, 27086, 8426, 8030, 9999, 3524, 4227, 4093, 4660, 204, 4394, 24597, 8083, 963, 2005, 9891, 15931, 10145, 8325, 3], 'rationale': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 'label': '2_hate', 'target_rationale': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'target_label': ['political']}
```

- *token_ids* is tokenized comment by KcBERT tokenizer include special token.
- *rationale* is binary list that indicates whether offensiveness rationale exists for each token
- *label* is transformed label.
  - *normal* : Comments with a value of 0 for all ratings.
  - *offensive* : Comments with a rating greater than 0 but not toward a protected group (gender, politics, etc.).
  - *1_hate (L1 hate)* : Comments with 1 as highest rating toward protected group. Additionally, offensive comments without a specified rationale for offensiveness.
  - *2_hate (L2 hate)* : Comments with 2 as highest rating and labeled spans for the offensiveness rationale.
- *target_rationale* is binary list that indicates whether target rationale exists for each token
- *target_label* is multi-label target label.
    - *Gender, Age, Race, Religion, Politics, Job, Disability, Individuals, Others*


### Label distribution of transformed abusive language categories across the split data
![label_distribution](https://github.com/ssu-humane/K-HATERS/assets/76468616/d08aa6df-923c-4fcf-88ae-c322d39acbed)<br>
We split the dataset of 192,158 samples into 172,158/10,000/10,000 for training, validation, and test purposes, ensuring the transformed label distribution is maintained.
<br>
## Code
### Training
```python
python train.py
```
&emsp; This trains the H+T model using transformed labels.

### Evaluation
```python
python evaluation.py
```
&emsp; This evaluates the trained H+T model using the test set.
