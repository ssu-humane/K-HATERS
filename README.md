# K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings

This repository provides the code and dataset for our EMNLP'23 findings paper.

Download the data [here](https://huggingface.co/datasets/humane-lab/K-HATERS/tree/main/transformed) and place it in each directory as follows.<br>
- *Total_data_4.pickle* -> Data
- *train_data.pickle*, *val_data.pickle*, *test_data.pickle* -> Data/Total_data_4<br>

### For training, run this code.
```python
python train.py
```
&emsp; This code trains the H+T model using transformed labels.

### For evaluation, run this code.
```python
python evaluation.py
```
&emsp; This code evaluates the trained H+T model using the test set.
