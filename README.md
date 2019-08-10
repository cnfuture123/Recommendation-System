# Recommender-System
Recommender System With Surprise

## Introduction
This project is to build a recommender system using Surprise which is a Python scikit building and analyzing recommender systems. And the designed recommender system is used to recommend similar movies for a given movie. The data is getting from MovieLens 100K Dataset: https://grouplens.org/datasets/movielens/100k/.

## Dataset Description
After download and unzip the "ml-100k.zip" file from above link, we can get a list of separate files which contain different contents. In this project, we use "u.data" and "u.item" files only. Here is a short description for these two files:
```
u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items. Each user has rated at least 20 movies. 
              Users and items are numbered consecutively from 1. This is a tab separated list of user id | item id | 
              rating | timestamp. The time stamps are unix seconds since 1/1/1970 UTC.
```
```
u.item     -- Information about the items (movies); this is a tab separated list of movie id | movie title | release date | 
              video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | 
              Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | 
              Western | The last 19 fields are the genres, a 1 indicates the movie is of that genre, a 0 indicates it is not;
              movies can be in several genres at once. The movie ids are the ones used in the u.data data set.
```

## Algorithms
Following algorithms are used in the experiment to find the best movie recommender system with miminum RMSE.
* KNNBasic
* KNNWithMeans
* KNNBaseline
* BaselineOnly
* NormalPredictor
* SVD
* SVDpp
* NMF

|      Alg            |     RMSE      |
| -------------       | ------------- |
|   KNNBasic          |    1.0073     |
|   KNNWithMeans      |    0.9343     |
|   KNNBaseline       |    0.9272     |
|   BaselineOnly      |    0.9442     |
|   NormalPredictor   |    1.5191     |
|   SVD               |    0.9481     |
|   SVDpp             |    0.9568     |
|   NMF               |    1.0027     |

## Model
Based on above experiment result, KNNBaseline algorithm can achieve best performance with minimum RMSE. Hence we use it to build model for recommender system.
```
Model:
      sim_options = {'name': 'pearson_baseline', 'user_based': False}
      trainset = data.build_full_trainset()
      alg = KNNBaseline(sim_options=sim_options)
      alg.fit(trainset)
```

