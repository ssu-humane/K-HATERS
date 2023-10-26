# K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings

This repository provides the code and dataset for our EMNLP'23 findings paper.

<!--
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
- *target_rationale* is binary list that indicates whether target rationale exists for each token
- *target_label* is multi-label target label.

### Label distribution of transformed abusive language categories across the split data

## Training
```python
python train.py
```
&emsp; This trains the H+T model using transformed labels.

## Evaluation
```python
python evaluation.py
```
&emsp; This evaluates the trained H+T model using the test set.
-->
