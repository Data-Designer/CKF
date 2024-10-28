<h1 align="center"> Collaborative Knowledge Fusion: A Novel Approach for Multi-task Recommender Systems via LLMs </h1>

## About Our Work

Update: 2024/10/28: We have created a repository for the paper titled *Collaborative Knowledge Fusion: A Novel Method for Multi-task Recommender Systems via LLMs*, which has been submitted to the *TKDE2024*. In this repository, we offer the original sample datasets, preprocessing scripts, and algorithm files to showcase the reproducibility of our work.

![image-20240120094651475](https://s2.loli.net/2024/01/20/DztabiuLphm4EOA.png)

## Requirements

- aiohttp==3.9.1
- altair==5.2.0
- black==23.11.0
- datasets==2.15.0
- jsonlines==4.0.0
- kiwisolver==1.4.5
- numpy==1.26.2
- pandas==2.1.3
- peft==0.4.0
- pyparsing==3.1.1
- scikit_learn==1.3.2
- torch==2.0.1
- tqdm==4.66.1
- transformers==4.31.0

## Data Sets

Owing to the copyright stipulations associated with the dataset, we are unable to provide direct upload access. However, it can be readily obtained by downloading directly from the official website: [AmazonDatasets](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/); [Movie-Lens Datasest](https://grouplens.org/datasets/movielens/)

The structure of the data set should be like,

```powershell
data
|_ source_data
|  |_ reviews_Books_5.json.gz
|  |_ meta_Books.json
|  |_ meta_Movies_and_TV.json.gz
|  |_ Movies_and_TV_5.json.gz
|  |_ movielen_train_ood2.pkl
|  |_ movielen_test_ood2.pkl

| |_ books_processed
| |_ movielen_processed
| |_ amazon_movie_processed

```

After processing, the structure of the data set should be like,

```powershell
data
|_ books_processed
|  |_ user_sequences_id.json
|  |_ user_rating.json
|  |_ user_review.json
|  |_ id_2_title.json
|  |_ user_sequences_title.json
|  |_ cold_user.json
|_ amazon_movie_processed
|  |_ user_sequences_id.json
|  |_ user_rating.json
|  |_ user_review.json
|  |_ id_2_title.json
|  |_ user_sequences_title.json
|  |_ cold_user.json
|_ |movielen_processed
|  |_ user_sequences_id.json
|  |_ user_rating.json
|  |_ user_review.json
|  |_ id_2_title.json
|  |_ user_sequences_title.json
|  |_ cold_user.json
```



## RUN

```powershell
# unzip all files into the data directory
cd /root/CKF/src
python generate_books_data.py #  generate books data for train
cd /root/CKF/src
python model_finetune_books.py # train file

cd /root/CKF/src
python evaluate.py ## evaluate file
```

## Contact

We will update the contact information and corresponding issue links after the review process is completed.
