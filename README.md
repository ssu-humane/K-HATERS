# K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific and Fine-Grained Ratings

This repository provides the code and dataset for our EMNLP'23 findings paper.

The link and instruction for its usage will be available very soon.
<br>
Download the data [here](https://huggingface.co/datasets/humane-lab/K-HATERS/tree/main/transformed) and place it in each directory as follows.<br>
- Total_data_4 -> Data
- train_data.pickle, val_data.pickle, test_data.pickle -> Data/Total_data_4<br>

### For training, run this code.
```python
python train.py bert True False False
```
### For evaluation, run this code.
```python
python evaluation.py bert True False False 0
```
