### Machine Learning for Public Policy
### Pipeline: Build, select, and evaluate classifiers
### Héctor Salvador López

import json
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import time

plt.style.use('ggplot')
OBJ_COL = 'amount_of_contract (constant USD) log'

# Classifiers to test
classifiers = {'LR': LogisticRegression(),
				'KNN': KNeighborsClassifier(),
				'DT': DecisionTreeClassifier(),
				'SVM': LinearSVC(),
				'RF': RandomForestClassifier(),
				'GB': GradientBoostingClassifier()}

grid = {#'LR': {'penalty': ['l1', 'l2'], 'C': [0.1, 1]}, 
		'LR': {'penalty': ['l1', 'l2'], 'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}, 
        'KNN': {'n_neighbors': [1, 5, 10, 25, 50, 100], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree']},
        #'KNN': {'n_neighbors': [5, 10], 'weights': ['uniform', 'distance'], 'algorithm': ['auto']},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, 20, 50, 100], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10]},
        'SVM' : {'C' : [0.1, 1]},
        'RF': {'n_estimators': [1, 10], 'max_depth': [1, 5, 10], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10]},
        'GB': {'n_estimators': [1, 10], 'learning_rate' : [0.1, 0.5], 'subsample' : [0.5, 1.0], 'max_depth': [1, 3, 5]},
        }


def classify(X, y, models, iters, threshold, metrics, top_perc):
	'''
	Takes:
		X, a dataframe of features 
		y, a dataframe of the label
		models, a list of strings indicating models to run (e.g. ['LR', 'DT'])

	Returns:
		A new dataframe comparing each classifier's performace on the given
		evaluation metrics.
	'''
	all_models = {}
	
	# for every classifier, try any possible combination of parameters on grid
	for index, clf in enumerate([classifiers[x] for x in models]):
		name = models[index]
		print(name)
		parameter_values = grid[name]
		all_models[name] = {}
		
		# run the model with all combinations of the above parameters
		for p in ParameterGrid(parameter_values):
			accuracy_per_iter = []
			precision_per_iter = []
			recall_per_iter = []
			f1_per_iter = []
			roc_auc_per_iter = []
			time_per_iter = []
			precision_top_n_p_iter = []
			avg_metrics = {}
			all_models[name][str(p)] = {}
			results = all_models[name][str(p)]
			clf.set_params(**p)

			# run iter number of iterations
			for i in range(iters): 

				# Construct train and test splits
				xtrain, xtest, ytrain, ytest = \
					train_test_split(X, y, test_size=0.2)
				
				try:
					start_time = time.time()
					# get the predicted results from the model
					if hasattr(clf, 'predict_proba'):
						yscores = clf.fit(xtrain,ytrain).predict_proba(xtest)[:,1]
					else:
						yscores = clf.fit(xtrain,ytrain).decision_function(xtest)
					
					xtest_temp = xtest.copy()
					xtest_temp['yscore'] = yscores
					xtest_temp['expected_value'] = xtest_temp.apply(lambda row: \
						row['yscore'] * row[OBJ_COL], axis=1)
					xtest_temp['y'] = ytest
					xtest_temp = xtest_temp.sort_values(by='expected_value', ascending=False)

					# To define number of contracts to investigate
					n = round(ytest.size * top_perc)

					yhat = np.asarray([1 if i >= threshold else 0 for i in xtest_temp['yscore']])
					end_time = time.time()

					# obtain metrics
					precision_top_n_p_iter.append(precision_score(xtest_temp['y'][:n], yhat[:n]))
					print(yhat[:n])
					print(xtest_temp['y'][:n])
					print(precision_score(xtest_temp['y'][:n], yhat[:n]))

					mtrs = evaluate_classifier(ytest, yhat)
					for met, value in mtrs.items():
						eval('{}_per_iter'.format(met)).append(value)
					time_per_iter.append(end_time - start_time)
					
				except IndexError:
					print('Error')
					continue

			# store average metrics of model p
			for met in metrics:
				avg_metrics[met] = np.mean(eval('{}_per_iter'.format(met)))
				results[met] = avg_metrics[met]
			results['time'] = np.mean(time_per_iter)
			results['precision_top_n'] = np.mean(precision_top_n_p_iter)

		print('Finished running {}'.format(name))

	# dump everything in a json for future reference
	with open('all_models.json', 'w') as fp:
		json.dump(all_models, fp)

	return all_models

#########################
##  check apply model  ##
#########################
def apply_model(winner, X):
	'''
	winner = (best_model, best_metric)
		best_model = model, params
	'''
	yhat = classifiers[winner[0][0]].predict(data)
	return yhat

def select_best_models(results, models, d_metric):
	columns = ['roc_auc', 'f1', 'precision', 'recall', 'time', 'parameters', 'precision_top_n']
	rv = pd.DataFrame(index = models, columns = columns)
	best_metric = 0
	best_model = 0
	best_models = {}

	for model, iters in results.items():
		print(model, iters)
		top_intra_metric = 0
		best_models[model] = {}
		for params, metrics in iters.items():
			header = [key for key in metrics.keys()]
			if metrics[d_metric] > top_intra_metric:
				top_intra_metric = metrics[d_metric]
				best_models[model]['parameters'] = params
				best_models[model]['metrics'] = metrics

		try:
			to_append = [value for value in best_models[model]['metrics'].values()]
			to_append.append(best_models[model]['parameters'])
		except:
			to_append = [0]
		
		rv.loc[model] = to_append
		if top_intra_metric > best_metric:
			best_metric = top_intra_metric
			best_model = model, params

	return rv, best_models, (best_model, best_metric)

def gen_precision_recall_plots(X, y, best_models):
	'''
	'''
	xtrain, xtest, ytrain, ytest = \
						train_test_split(X, y, test_size=0.2, random_state=0)

	for name, d in best_models.items():
		clf = classifiers[name]
		p = eval(d['parameters'])
		clf.set_params(**p)
		y_true = ytest
		if hasattr(clf, 'predict_proba'):
			y_prob = clf.fit(xtrain, ytrain).predict_proba(xtest)[:,1]
		else:
			y_prob = clf.fit(xtrain, ytrain).decision_function(xtest)
		plot_precision_recall_n(y_true, y_prob, name)

def evaluate_classifier(ytest, yhat):
	'''
	For an index of a given classifier, evaluate it by various metrics
	'''
	# Metrics to evaluate
	metrics = {'precision': precision_score(ytest, yhat),
				'recall': recall_score(ytest, yhat),
				'f1': f1_score(ytest, yhat),
				'roc_auc': roc_auc_score(ytest, yhat),
				'accuracy': accuracy_score(ytest, yhat)}
	
	return metrics

def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    '''
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

