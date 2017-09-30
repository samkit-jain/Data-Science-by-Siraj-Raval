from tpot import TPOTRegressor
from sklearn.cross_validation import train_test_split
import pandas as pd 
import numpy as np
import math


# clean the data
def clean_data(temperatures):
	# consider values from India only
	temperatures = temperatures[temperatures['Country'] == 'India']
	
	# split date to year, month and day
	temperatures['y'] = temperatures['dt'].apply(lambda x: str(x).split('-')[0])
	temperatures['m'] = temperatures['dt'].apply(lambda x: str(x).split('-')[1])
	temperatures['d'] = temperatures['dt'].apply(lambda x: str(x).split('-')[2])

	# remove NaN values
	temperatures = temperatures[np.isfinite(temperatures['AverageTemperature'])]
	
	# remove date column
	del temperatures['dt']
	del temperatures['Country']

	# rename AverageTemperature to Class
	temperatures.rename(columns={'AverageTemperature': 'Class'}, inplace=True)
	
	return temperatures


# Load the data
temperatures = pd.read_csv('GlobalLandTemperaturesByCountry.csv', usecols=[0, 1, 3])

temperatures = clean_data(temperatures)

# store classes
temperatures_class = temperatures['Class'].values

# split training, testing, and validation data
training_indices, validation_indices = training_indices, testing_indices = train_test_split(list(temperatures.index), train_size=0.8, test_size=0.2)

# let Genetic Programming find best ML model and hyperparameters
tpot = TPOTRegressor(generations=5, verbosity=2)
tpot.fit(temperatures.drop('Class', axis=1).loc[training_indices].values, temperatures.loc[training_indices, 'Class'].values)

# score the accuracy
print("Accuracy:", tpot.score(temperatures.drop('Class', axis=1).loc[validation_indices].values, temperatures.loc[validation_indices, 'Class'].values))

# export the generated code
tpot.export('pipeline.py')
