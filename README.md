# K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings

This repository introduces K-HATERS, the largest hate speech detection corpus in Korean to date, shared along with our EMNLP'23 findings paper.

## Data sources

- We collected 191,633 comments from 52,590 news articles that were published through Naver News over two months in 2021.
- We included 8,367 comments from the labeled set of an available [corpus](https://github.com/kocohub/korean-hate-speech).

All comments were annotated through CashMission, a crowdsourcing service run by SELECTSTAR.

## Data distribution

We provide the split data of 172,158/10,000/10,000 for training, validation, and test purposes, ensuring the label distribution is maintained.

![label_distribution](https://github.com/ssu-humane/K-HATERS/assets/76468616/d08aa6df-923c-4fcf-88ae-c322d39acbed)<br>

## Data description

```python
>>> data['train'][42]
{'text': '군대도 안간 놈 이 주둥아리 는 씽씽하네..보수 놈 들..군대는 안가고 애국이냐..#@이름#,#@이름#,', 'label': '1_hate', 'target_label': ['political'], 'offensiveness_rationale': [[7, 8], [11, 15], [27, 28]], 'target_rationale': [[24, 26], [46, 51], [52, 57]]}
```

- Abusive language categories (```label```)
  + *L2_hate*: Comments with explicit forms of hate expressions toward one of the groups of protected attributes (e.g., gender, age, race, ...)
  + *L1_hate*: Comments with more implicit forms of hate expressions
  + *Offensive*: Comments that express offensiveness but not toward a protected attribute group
  + *Normal*: The rest comments
- Multi-label target categories (```target_label```): list of offensiveness targets. A comment can have zero or multiple targets.
  + List of target categories: gender, age, race, religion, politics, job, disability, individuals, and others.
- Annotators' rationales for the strength of ratings (```offensiveness_rationale```): lists providing annotators' rationales for the strength of ratings. The list includes the start and end indices of highlight spans.
- Annotators' rationales for the target of offensiveness (```target_rationale```)

## How to use it
The dataset is available through the HuggingFace Hub. 

```python
from datasets import load_dataset

data = load_dataset('humane-lab/K-HATERS')
train = data['train']
valid = data['valid']
test =  data['test']
```

## Where are the ratings?
The original dataset was labeled by annotators for thirteen rating variables on a 3-point Likert scale, each indicating the offensiveness strength toward a target.
In this paper, we proposed a label transformation scheme leading to 4-class abusive language categories and multi-label target labels for efficient modeling.
The above code is to access the transformed dataset. Please refer to the details of the labeling scheme in our paper.

If you want to use the dataset with rating variables, please refer to the following code.
```python
from datasets import load_dataset

data = load_dataset('humane-lab/K-HATERS-Ratings')
```

## Acknowledgements

The dataset construction was supported by DATUMO (SELECTSTAR) through the "2022 AI Dataset Supporting Business" program.

## License and Further Usages

This dataset is shared under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en).
According to this license, you are free to use the dataset as long as you provide appropriate attribution (e.g., citing our paper) and share any derivative works under the same license.

```bibtex
@article{park2023haters,
  title={K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings},
  author={Park, Chaewon and Kim, Suhwan and Park, Kyubyong and Park, Kunwoo},
  journal={Findings of the EMNLP 2023},
  year={2023}
}
```



