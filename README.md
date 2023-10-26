# K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings

This repository introduces K-HATERS, the largest hate speech detection corpus in Korean, shared along with our EMNLP'23 findings paper.

## How to use the dataset
The dataset is available through the HuggingFace Hub. 

```python
from datasets import load_dataset

data = load_dataset('humane-lab/K-HATERS')
```
```python
>>> data['train'][0]
{'comment': '예시 커멘트', 'label': '2_hate', 'target_label': ['political']}
```

- *token_ids* is tokenized comment by KcBERT tokenizer include special token.
- *rationale* is binary list that indicates whether offensiveness rationale exists for each token
- *label* is transformed label.
  - *normal, offensive, 1_hate, 2_hate*
- *target_rationale* is binary list that indicates whether target rationale exists for each token
- *target_label* is multi-label target label.
  - *Gender, Age, Race, Religion, Politics, Job, Disability, Individuals, Others*

## Label description
- *2_hate (L2 hate)* : Comments with 2 as highest rating and labeled spans for the offensiveness rationale.
- *1_hate (L1 hate)* : Comments with 1 as highest rating toward protected group. Additionally, offensive comments without a specified rationale for offensiveness.
- *normal* : Comments with a value of 0 for all ratings. 
- *offensive* : Comments with a rating greater than 0 but not toward a protected group (gender, politics, etc.).

  
### Label distribution of transformed abusive language categories across the split data
![label_distribution](https://github.com/ssu-humane/K-HATERS/assets/76468616/d08aa6df-923c-4fcf-88ae-c322d39acbed)<br>
- We split the dataset of 192,158 samples into 172,158/10,000/10,000 for training, validation, and test purposes, ensuring the transformed label distribution is maintained.

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
