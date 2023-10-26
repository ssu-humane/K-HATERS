# K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings

This repository introduces K-HATERS, the largest hate speech detection corpus in Korean, shared along with our EMNLP'23 findings paper.

## Dataset distribution

We provide the split data of 172,158/10,000/10,000 for training, validation, and test purposes, ensuring the label distribution is maintained.

![label_distribution](https://github.com/ssu-humane/K-HATERS/assets/76468616/d08aa6df-923c-4fcf-88ae-c322d39acbed)<br>

## Label descriptions
- *L2_hate*: Comments with explicit forms of hate expressions toward one of the groups of protected attributes (e.g., gender, age, race, ...)
- *L1_hate*: Comments with more implicit forms of hate expressions
- *Offensive*: Comments that express offensiveness but not toward a protected attribute group
- *Normal*: Comments with a value of 0 for all ratings 

## How to use the dataset
The dataset is available through the HuggingFace Hub. 

```python
from datasets import load_dataset

data = load_dataset('humane-lab/K-HATERS')
```
```python
>>> data['train'][0]
{'text': '예시 커멘트', 'label': '2_hate', 'target_label': ['political'], 'offensiveness_rationale': [(start1, end1),(start2, end2)], 'target_rationale': [(start1,end1)]}
```

- *text*: news comments
- *label*: 4-class abusive language categories (L2_hate, L1_hate, Offensive, Normal))
- *target*: multi-label target categories (Gender, Age, Race, Religion, Politics, Job, Disability, Individuals, Others)
- *offensivenes_rationale*: lists providing annotators' rationales for the strength of ratings. The list includes the start and end indices of highlight spans.
- *target_rationale*: annotators' rationales for the target of offensiveness

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
