# In this code, 'dataset' should be ratings(K-HATERS-Ratings) dataset
def labelTransformation(dataset):
    t_dataset = []
    
    for data in dataset:
        data = list(data.values()) # [Text, Gender, Age, Disability, Job, Race, Religion, Politics, Individuals, Other, Insult, Swear word, Threat, Obscenty, Target labels, Offensiveness rationale, Target rationale]
        rationale = data[15]
        GRP = data[1:8]
        ratings = data[1:14]

        if sum(ratings)==0:
            t_label = 'normal'
        elif sum(GRP)==0:
            t_label = 'offensive'
        elif sum(rationale)!=0 and max(ratings)>1:
            t_label = '2_hate'
        else:
            t_label = '1_hate'
        t_dataset.append((data[0],data[1],t_label,data[15],data[16]))
    return t_dataset
