### Machine Learning for Public Policy
### Pipeline: Explore
### Héctor Salvador López

import pandas as pd
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

def basics(data):
	print('Observations:\n' + '{}\n'.format(data.shape[0]))
	print('{} features:'.format(data.shape[1]))
	i = 1
	for key in data.keys():
		print('    {}) {}'.format(i, key))
		i += 1
	print('\n')
	print('Sample observations:\n' + '{}\n'.format(data.head()))

def desc_statistics(data):
	'''
	Takes:
		data, a pd.dataframe 

	Prints:
		keys of the df
		first five observations
		number of observations 
		descriptive statistics
	'''
	summary = data.describe().T
	summary['median'] = data.median()
	summary['skew'] = data.skew() # skew of normal dist = 0
	summary['kurtosis'] = data.kurt() # kurtosis of normal dist = 0
	summary['missing_vals'] = data.count().max() - data.describe().T['count']
	print('Descriptive statistics:\n' + '{}\n'.format(summary.T))

def corr(data):
	print('Correlation matrix:\n' + '{}\n'.format(data.corr()))


def plots(data):
	'''
	Takes:
		data, a pd.dataframe 

	Generates histograms in a separate folder
	'''
	print('Check the current folder for default histograms of these features.')
	for feature in data.keys():
		unique_vals = len(data[feature].value_counts())
		figure = plt.figure()
		if unique_vals == 1:
			data.groupby(feature).size().plot(kind='bar')
		elif unique_vals < 15:
			bins = unique_vals
			data[feature].hist(xlabelsize=10, ylabelsize=10, bins=unique_vals)
		else:
			data[feature].plot.hist()

		# details for plots
		plt.ylabel('Frequency')
		plt.title('{}'.format(feature))
		plt.savefig('histograms/{}'.format(feature) + '_hist')
		plt.close()


def crosstabs(data, label, features):
	'''
	Takes:
		data, a pd.dataframe
		categorical, an int indicating a label
		covariates, a list of strings with the features

	Prints crosstabs of desired features.
	'''
	keys = [i for i in data.keys()]
	for feature in features:
		print('Crosstab table for {} and {}:'.format(label, feature))
		print('{}'.format(pd.crosstab(data[label], data[feature])) + '\n')


		