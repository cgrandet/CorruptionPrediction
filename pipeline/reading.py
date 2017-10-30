### Machine Learning for Public Policy
### Pipeline: Reading
### Héctor Salvador López

import csv
import json
import pandas as pd 


# read file depending on the data type
def read(filename, data_type = 'csv'):

	data_types_all = ['csv', 'json']
	assert data_type in data_types_all

	if data_type == 'csv':
		df = read_csv(filename)
	elif data_type == 'json':
		df = read_json(filename)
	return df


def read_csv(filename):
	df = pd.read_csv(filename)
	return df

# def read_json(filename):
# 	with open(filename) as data:
# 		df = json.load(data) 
# 	return df