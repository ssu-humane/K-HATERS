# K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings

This repository provides the code and dataset for our EMNLP'23 findings paper.

<!--Download the data [here](https://huggingface.co/datasets/humane-lab/K-HATERS/tree/main/transformed) and place it in each directory as follows.<br>
- *Total_data_4.pickle* -> Data
- *train_data.pickle*, *val_data.pickle*, *test_data.pickle* -> Data/Total_data_4<br>

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
### How to use dataset throught Hugging Face datasets library
```python
from datasets import load_dataset

data = load_dataset('humane-lab/K-HATERS')
```
