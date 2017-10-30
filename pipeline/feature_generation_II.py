import pandas as pd 

'''
We plan to expand upon that work and create features in 5 categories:
Contract amount, procurement type, sector, supplier and borrower_country_std characteristics.
Here is a list of the features we have generated:

Contract amount:
	Contract larger than average borrower_country_std contract all years - Dummy
	Contract larger than average borrower_country_std contract specific year - Dummy
Contract larger than average sector contract all years - Dummy
	Contract larger than average sector contract specific year - Dummy
Contract larger than average supplier contract all years - Dummy
	Contract larger than average supplier contract specific year - Dummy
	Total value of contracts for borrower_country_std in the year of the contract - Float
	Ratio of contract amount over total amount per borrower_country_std - Float
Ratio of contract amount over total amount per sector - Float
Ratio of contract amount over total amount per supplier  - Float
	
Procurement type: 
	Percentage of contracts obtained through international bidding by borrower_country_std - Float
	Percentage of contracts obtained through international bidding by supplier - Float
	Percentage of contracts obtained through international bidding by sector - Float
	Percentage of contracts obtained through single source selection by borrower_country_std - Float
	Percentage of contracts obtained through single source selection by supplier - Float
	Percentage of contracts obtained through  single source selection by sector - Float
	Contract obtained through a method not present in 90% of the contracts - Dummy
Contract obtained through a method not present in 90% of the contracts by borrower_country_std- Dummy
Contract obtained through a method not present in 90% of the contracts by supplier- Dummy
Contract obtained through a method not present in 90% of the contracts by sector- Dummy
	
Sector:
	Contract belonging to Public Works - Dummy
	Contract belonging to Pharmaceutical acquisition - Dummy
	Contract belonging to Consultant services - Dummy 
	
Additionally, we plan to generate the following features 
	
Supplier: 
	Rate of concentration of projects for specific supplier - Float
	Supplier for a different borrower_country_std than project - Dummy
	Atomization of supplier (supplier contract amount divided by project amount) - Float
	Diversification of supplier (how many sectors is he a contractor for) - Integer
	Diversification of supplier (how many countries is he a contractor for) - Integer
	Diversification of supplier (how many projects is he a contractor for) - Integer
	
borrower_country_std characteristics:
	Corruption index from Transparency International - Float 
	Concentration of resources by year - Float
	Abrupt change in concentration of resources from one year to another - Dummy
	Abrupt change in number of contracts from one year to another - Dummy
'''


def generate_discrete_variable(data, criteria_dict):
	'''
	Write a sample function that can discretize a continuous variable 
	Input 
	data: a Pandas dataframe
	criteria_dict: a dictionary where the keys are variables from data
	and the values is a list of n elements that you want to discretize.
	Each n element is a tuple with the value of the new discrete variable
	and the range of values from the continous variable.

	i.e. 
	CRITERIA = {"DebtRatio":[("less than 50%",0,.5),("less than 100%",5,1),
	("less than 5 times",1,5),("less than 10 times",5,10),
	("more than 10 times",10,float("inf"))]}

	'''
	#Generate categorical values from continous variable 
	for column, criteria in criteria_dict.items():
		#The parameter list contains the labels
		parameter_list = []
		#The range set contains the values for the different parameters
		range_set = set()
		for parameter in range(len(criteria)):
			parameter_list.append(criteria[parameter][0])
			range_set.add(criteria[parameter][1])
			range_set.add(criteria[parameter][2])

		range_list = list(range_set)
		range_list.sort()

		#Generate categorical variables, the "right" option
		#creates set [a,b) to satisfy greater or equal restriction
		#for lower limit. 
		data[column] = pd.cut(data[column],range_list,
						right = False, labels = parameter_list)
		
		#Drop rows that did not have a categorical match
		data = data[~data[column].isnull()]

	return data 


def generate_continous_variable(data, variable_list):
	'''
	function that can take a categorical variable and create 
	a numerical variable from it

	The function will transform any string categorical variable into
	a numerical one. It makes more sense when is a categorical
	variable with only two categories to transform it into a dummy
	variable.

	Input:
	data: A Pandas dataframe
	variable_list: a list of variables names in the data
	'''
	for variable in variable_list:
		if data[variable].dtype == "object":
			list_values = list(data.groupby(variable).groups.keys())
			for i,value in enumerate(list_values):
				data[variable] = data[variable].replace(value,i)

	return data 


def is_variable_above_sd(data, name, threshold):
	sd = data[name].std()
	mean = data[name].mean()
	binarizer = preprocessing.Binarizer(threshold= mean + (sd*threshold))
	data[name+"above_sd"] = binarizer.transform(X)
	
def logarithmic_transformation(data, name):
	transformer = FunctionTransformer(np.log1p)
	data[name+"log"] = transformer.transform(data[name])


def polynomial_transformation(data, polynomial, name, exponent):
	poly = PolynomialFeatures(exponent)
	data[name+"exponent"] = poly.fit_transform(data[name])   

def normalize_minmax(data, name):
	min_max_scaler = preprocessing.MinMaxScaler()
	data[name + "min_max"] = min_max_scaler.fit_transform(data[name])

def normalize_standard(data,name):
	data[name + "normal"] = preprocessing.scale(data[name])

def normalize_max(data, name):
	max_value = data[name].max 
	data[name+"max_standard"] = data[name] / max_value

def agg_by_group(data,name,group,func,new_variable_name):
	agg = data.groupby(group)[name].transform(func)
	data[new_variable_name] = agg

def larger_than_average(data, name, new_variable_name):
	mean = data[name].mean()
	data[new_variable_name] = data[name] > mean 

def monotonic_growth(data, variable_list, new_variable_name):
	is_increasing = (data[variable_list].T.diff().fillna(0) >= 0).all()
	data[new_variable_name] = is_decreasing

def monotonic_decrease(data, variable_list, new_variable_name):
	is_decreasing = (data[variable_list].T.diff().fillna(0) <= 0).all()
	data[new_variable_name] = is_decreasing

def consistent_high(data, variable_list, new_variable_name):
	high = (data[variable_list] > (data[variable_list].mean(axis =1) + data[variable_list].std(axis =1))).all(axis = 1)
	data[new_variable_name] = high

def consistent_low(data, variable_list, new_variable_name):
	low = (data[variable_list] < (data[variable_list].mean(axis =1) + data[variable_list].std(axis =1))).all( axis = 1)
	data[new_variable_name] = low

def anomaly_increase(data,variable_list,new_variable_name):
	anomaly = (data[variable_list].T.diff().T > (data[variable_list].T.diff().mean(adatais = 1)+ 2*data[variable_list].T.diff().std(adatais = 1))).any(adatais = 1)
	data[new_variable_name] = anomaly

def anomaly_decrease(data,variable_list,new_variable_name):
	anomaly = (data[variable_list].T.diff().T < (data[variable_list].T.diff().mean(adatais = 1)+ 2*data[variable_list].T.diff().std(adatais = 1))).any(adatais = 1)
	data[new_variable_name] = anomaly

def larger_than_group_average(data,variable,group,function,new_variable_name):
	larger = data[variable] > (data[variable].groupby(data[group]).transform(function))
	data[new_variable_name] = larger
	data[new_variable_name].replace("True", 1)
	data[new_variable_name].replace("False", 1)


def stats_per_group(data,variable,group,function,new_variable_name):
	larger = data[variable].groupby(data[group]).transform(function)
	data[new_variable_name] = larger
	data[new_variable_name].replace("True", 1)
	data[new_variable_name].replace("False", 1)

def ratio_per_group(data,variable,group,new_variable_name):
	larger = (data[variable] / data[variable].groupby(data[group]).transform("sum"))*100
	data[new_variable_name] = larger


def percentage_per_categorical_variable(data,variable,group):
	list_variable = list(data[variable].unique())
	for i in list_variable:
		data[i]= data[variable] == i 
		percentage = data[i].groupby(data[group]).transform("sum")/ data[variable].groupby(data[group]).transform("count")
		data[i+group+"percentage"] = percentage
	

def ratio_total(data,variable,group, new_variable_name):
	ratio = data[variable].groupby(data[group]).transform("sum") / data[variable].sum()
	data[new_variable_name] = ratio


def supplier_different(data,variable1, variable2,new_variable_name):
	data[new_variable_name] = data[variable1] == data[variable2]


def atomization(data,variable,group_list,group_den, new_variable_name):
	num = data.groupby(group_list)[variable].transform("sum") 
	den = data[variable].groupby(data[group_den]).transform("sum")
	ratio = num / den
	data[new_variable_name] = ratio


def diversification(data, group, variable, new_variable_name ):
	f = lambda x: x.nunique()
	diver = data.groupby(group)[variable].transform(f)
	data[new_variable_name] = diver


if __name__ == '__main__':
	
	data = pd.read_csv("data/tothecheck.csv")

	agg_by_group(data,"amount_of contract (PPP)","resolved_supplier","sum","resolved_supplier_sum")
	agg_by_group(data,"amount_of contract (PPP)","resolved_supplier","mean","resolved_supplier_mean")
	agg_by_group(data,"amount_of contract (PPP)","resolved_supplier","std","resolved_supplier_std")
	agg_by_group(data,"project_total_amount (PPP)","resolved_supplier","sum","resolved_supplier_sum_project")
	agg_by_group(data,"project_total_amount (PPP)","resolved_supplier","mean","resolved_supplier_mean_project")
	agg_by_group(data,"project_total_amount (PPP)","resolved_supplier","std","resolved_supplier_std_project")

	agg_by_group(data,"amount_of contract (PPP)","project_name","sum", "project_name_sum")
	agg_by_group(data,"amount_of contract (PPP)","project_name","mean","project_name_mean")
	agg_by_group(data,"amount_of contract (PPP)","project_name","std","project_name_std")
	agg_by_group(data,"project_total_amount (PPP)","project_name","sum","project_name_sum_project")
	agg_by_group(data,"project_total_amount (PPP)","project_name","mean","project_name_mean_project")
	agg_by_group(data,"project_total_amount (PPP)","project_name","std","project_name_std_project")

	# agg_by_group(data,"amount_of contract (PPP)","wb_contract_number","sum","wb_contract_number_sum")
	# agg_by_group(data,"amount_of contract (PPP)","wb_contract_number","mean","wb_contract_number_mean")
	# agg_by_group(data,"amount_of contract (PPP)","wb_contract_number","std","wb_contract_number_std")
	# agg_by_group(data,"project_total_amount (PPP)","wb_contract_number","sum","wb_contract_number_sum_project")
	# agg_by_group(data,"project_total_amount (PPP)","wb_contract_number","mean","wb_contract_number_mean_project")
	# agg_by_group(data,"project_total_amount (PPP)","wb_contract_number","std","wb_contract_number_std_project")


	larger_than_average(data,"amount_of contract (PPP)", "amount_of contract (PPP)+larger_average")
	larger_than_average(data,"project_total_amount (PPP)", "project_total_amount (PPP)+larger_average")
	larger_than_average(data,"resolved_supplier_sum", "resolved_supplier_sum+larger_average")
	larger_than_average(data,"resolved_supplier_sum_project", "resolved_supplier_sum_project+larger_average")


	larger_than_group_average(data,"amount_of contract (PPP)","borrower_country_std","mean","by_borrower_country_std_amount")
	larger_than_group_average(data,"amount_of contract (PPP)", "major_sector", "mean", "by_sector_amount")
	larger_than_group_average(data,"amount_of contract (PPP)", "fiscal_year", "mean", "by_year_amount")
	larger_than_group_average(data,"amount_of contract (PPP)", "procurement_category", "mean", "by_cat_amount")
	larger_than_group_average(data,"amount_of contract (PPP)", "procurement_method", "mean", "by_meth_amount")
	larger_than_group_average(data,"amount_of contract (PPP)", "procurement_type", "mean", "by_type_amount")
	larger_than_group_average(data,"amount_of contract (PPP)", "region", "mean", "by_year_amount")

	larger_than_group_average(data,"resolved_supplier_sum","borrower_country_std","mean","by_borrower_country_std_resolved_supplier")
	larger_than_group_average(data,"resolved_supplier_sum", "major_sector", "mean", "by_sector_resolved_supplier")
	larger_than_group_average(data,"resolved_supplier_sum", "fiscal_year", "mean", "by_year_resolved_supplier")
	larger_than_group_average(data,"resolved_supplier_sum", "procurement_category", "mean", "by_cat_resolved_supplier")
	larger_than_group_average(data,"resolved_supplier_sum", "procurement_method", "mean", "by_meth_resolved_supplier")
	larger_than_group_average(data,"resolved_supplier_sum", "procurement_type", "mean", "by_type_resolved_supplier")
	larger_than_group_average(data,"resolved_supplier_sum", "region", "mean", "by_year_resolved_supplier")

	ratio_per_group(data,"amount_of contract (PPP)","borrower_country_std", "ratio_by_borrower_country_std_amount")
	ratio_per_group(data,"amount_of contract (PPP)", "major_sector","ratio_by_sector_amount")
	ratio_per_group(data,"amount_of contract (PPP)", "fiscal_year", "ratio_by_year_amount")
	ratio_per_group(data,"amount_of contract (PPP)", "procurement_category", "ratio_by_cat_amount")
	ratio_per_group(data,"amount_of contract (PPP)", "procurement_method", "ratio_by_meth_amount")
	ratio_per_group(data,"amount_of contract (PPP)", "procurement_type", "ratio_by_type_amount")
	ratio_per_group(data,"amount_of contract (PPP)", "region", "ratio_by_year_amount")
	ratio_per_group(data,"amount_of contract (PPP)", "supplier_country", "ratio_by_resolved_supplierc_amount")

	percentage_per_categorical_variable(data,"procurement_method","borrower_country_std")
	percentage_per_categorical_variable(data,"procurement_method","major_sector")
	percentage_per_categorical_variable(data,"procurement_method","fiscal_year")
	percentage_per_categorical_variable(data,"procurement_method","resolved_supplier")
	percentage_per_categorical_variable(data,"procurement_category","borrower_country_std")
	percentage_per_categorical_variable(data,"procurement_category","major_sector")
	percentage_per_categorical_variable(data,"procurement_category","fiscal_year")
	percentage_per_categorical_variable(data,"procurement_category","resolved_supplier")
	percentage_per_categorical_variable(data,"major_sector","resolved_supplier")

	supplier_different(data,"borrower_country_std", "supplier_country","resolved_supplier_different_borrower_country_std")

	ratio_total(data,"amount_of contract (PPP)","resolved_supplier","rate_resolved_supplier_total")
	ratio_total(data,"project_total_amount (PPP)","resolved_supplier","rate_resolved_supplier_total")
	ratio_total(data,"amount_of contract (PPP)","supplier_country","rate_resolved_supplierc_total")
	ratio_total(data,"project_total_amount (PPP)","supplier_country","rate_resolved_supplierc_total")
	ratio_total(data,"amount_of contract (PPP)","borrower_country_std","rate_borrower_country_std_total")
	ratio_total(data,"amount_of contract (PPP)","resolved_supplier","rate_fiscal_total")
	ratio_total(data,"amount_of contract (PPP)","project_name","rate_resolved_supplier_total")

	atomization(data,"amount_of contract (PPP)",["resolved_supplier","project_name"] ,"resolved_supplier", "atomization+resolved_supplier+contract")
	atomization(data,"project_total_amount (PPP)",["resolved_supplier","project_name"],"resolved_supplier", "atomization+resolved_supplier+project")
	atomization(data,"amount_of contract (PPP)",["major_sector","project_name"] ,"major_sector", "atomization+sector+contract")
	atomization(data,"project_total_amount (PPP)",["major_sector","project_name"],"major_sector", "atomization+sector+project")
	atomization(data,"amount_of contract (PPP)",["procurement_method","project_name"],"procurement_method", "atomization+procurement+project")
	atomization(data,"project_total_amount (PPP)",["procurement_method","project_name"],"procurement_method", "atomization+procurement+project")


	diversification(data,"resolved_supplier","major_sector","diversification_resolved_supplier_major_sector")
	diversification(data,"resolved_supplier","borrower_country_std","diversification_resolved_supplier_country")
	diversification(data,"resolved_supplier","project_name","diversification_resolved_supplier_project_name")
	diversification(data,"supplier_country","major_sector","diversification_resolved_supplierc_major_sector")
	diversification(data,"supplier_country","borrower_country_std","diversification_resolved_supplierc_borrower_country_std")
	diversification(data,"supplier_country","project_name","diversification_resolved_supplierc_project_name")

	data.to_csv("data/tothecheck_II.csv")