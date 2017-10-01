import csv
import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_stackexchange
from lightfm.evaluation import auc_score


def sample_recommendation(model, data, user_ids):
	# number of users and questions in training data
	n_users, n_items = data['train'].shape
	result = ""

	# generate recommendations for each user we input
	for user_id in user_ids:
		# labels of questions they answered
		# questions they answered = data['item_features'][user_id,:]
		known_positives = data['item_feature_labels'][data['item_features'].tocsr()[user_id].indices]

		# questions our model predicts they will answer
		scores = model.predict(user_id, np.arange(n_items), item_features=data['item_features'])
		# np.argsort(-scores)
		
		# labels of questions they should answer
		top_items = data['item_feature_labels'][data['item_features'][np.argsort(-scores)].indices]
		
		# print out the results
		result += "\nUser %s" % user_id
		result += "\n     Labels of the questions they answered:"

		for x in known_positives[:3]:
			result += "\n        %s" % x

		result += "\n     Labels of the questions they should answer:"

		for x in top_items[:3]:
			result += "\n        %s" % x

	return result


data = fetch_stackexchange('crossvalidated', test_set_fraction=0.1, indicator_features=False, tag_features=True)

models = {
	'warp': LightFM(loss='warp', item_alpha=1e-6, no_components=3),
	'logistic': LightFM(loss='logistic', item_alpha=1e-6, no_components=3),
	'bpr': LightFM(loss='bpr', item_alpha=1e-6, no_components=3),
}


for name, model in models.items():
	print("---%s---" % name)
	model.fit(data['train'], item_features=data['item_features'], epochs=30, num_threads=2)

	# Whats' AUC? https://stackoverflow.com/questions/45451161/evaluating-the-lightfm-recommendation-model/45466481#45466481

	test_auc = auc_score(model, data['test'], train_interactions=data['train'], item_features=data['item_features'], num_threads=2).mean() 
	print("Test AUC: %.3f%%" % (test_auc * 100))
	
	# using same dict for storing auc and recommendations
	models[name] = {
		"auc": test_auc,
		"result": sample_recommendation(model, data, [69, 789, 3210]),
	}


max_auc = max([value["auc"] for value in models.values()])
name = [name for name, value in models.items() if value["auc"] == max_auc][0]

print("Highest AUC of %.3f%% attained by %s." % (max_auc, name))
print()
print("Result is")
print(models[name]["result"])
