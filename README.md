# K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific and Fine-Grained Ratings

This repository provides the code and dataset for our EMNLP'23 findings paper.

The link and instruction for its usage will be available very soon.

download data [here](https://huggingface.co/datasets/humane-lab/K-HATERS/tree/main/transformed)
Total_data_4 -> Data
train_data.pickle, val_data.pickle, test_data.pickle -> Data/Total_data_4

for training
```python
python train.py bert True False False
```
for evaluation
```python
python evaluation.py bert True False False 0
```
