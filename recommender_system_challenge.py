import csv
import io
import numpy as np
import pandas as pd
import zipfile
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k


def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape
    result = ""

    # generate recommendations for each user we input
    for user_id in user_ids:
        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        result += "\nUser %s" % user_id
        result += "\n     Known positives:"

        for x in known_positives[:3]:
            result += "\n        %s" % x

        result += "\n     Recommended:"

        for x in top_items[:3]:
            result += "\n        %s" % x

    return result


data = fetch_movielens(min_rating=4.0)

models = {
    'warp': LightFM(loss='warp'),
    'logistic': LightFM(loss='logistic'),
    'bpr': LightFM(loss='bpr'),
}


for name, model in models.items():
    print("---%s---" % name)
    model.fit(data['train'], epochs=30, num_threads=2)
    precision = precision_at_k(model, data['test']).mean()
    precision = precision * 100
    
    print("Test precision: %.3f%%" % precision)
    
    # using same dict for storing precision and recommendations
    models[name] = {
        "precision": precision,
        "result": sample_recommendation(model, data, [3, 25, 450]),
    }

max_precision = max([value["precision"] for value in models.values()])
name = [name for name, value in models.items() if value["precision"] == max_precision][0]

print("Highest precision of %.3f%% attained by %s." % (max_precision, name))
print("Result is")
print(models[name]["result"])
