import pandas as pd
import csv

#input file of training data
train_data = pd.read_csv('train.csv')

# data and target values are taken in separate dataframe
data = train_data[train_data.columns[:6]]

target = train_data.iloc[:,-1]

#no of positive and negative instances are calculated
no_positive, no_negative = target.value_counts()[1], target.value_counts()[0]

#probability of positive and negative is calculated
prob_positive = no_positive/len(target)
prob_negative = no_negative/len(target)

# data and target values are taken in separate dataframe
test_data = pd.read_csv('test.csv')

# data and target values are taken in separate dataframe
test_class = test_data.iloc[:,-1]

test_data = test_data[test_data.columns[:6]]

# function returns total negative values for any feature value
def calculate_negative(feature):
	feature_negative = {}

	for i in range(feature.nunique()):
		feature_negative[i+1] = 0

	for i in range(len(target)):
		if target[i] == 'negative':
			for j in range(feature.nunique()):
				if feature[i] == j+1:
					feature_negative[j+1] += 1

	return feature_negative

# function returns total positive values for any feature value
def calculate_positive(feature):
	feature_positive = {}
	
	for i in range(feature.nunique()):
		feature_positive[i+1] = 0

	for i in range(len(target)):
		if target[i] == 'positive':
			for j in range(feature.nunique()):
				if feature[i] == j+1:
					feature_positive[j+1] += 1
				
	return feature_positive

final_pos = {}
final_neg = {}
for i in range(6):
	final_pos[i] = calculate_positive(train_data.iloc[:,i])
	final_neg[i] = calculate_negative(train_data.iloc[:,i])

pred_val = []
error = 0
for index, rows in test_data.iterrows():
	# calculate probability using m-estimate equation for class positive. where 1 is added directly by taking in account Laplace smoothing. the same thing is done for all the
	# features of dataset
	x1 = ((final_pos[0][rows[0]]  ) + 1) / (train_data.iloc[:,0].nunique() + no_positive)

	x2 = ((final_pos[1][rows[1]]  ) + 1) / (train_data.iloc[:,1].nunique() + no_positive)

	x3 = ((final_pos[2][rows[2]] ) + 1) / (train_data.iloc[:,2].nunique() + no_positive)

	x4 = ((final_pos[3][rows[3]] ) + 1) / (train_data.iloc[:,3].nunique() + no_positive)

	x5 = ((final_pos[4][rows[4]] ) + 1) / (train_data.iloc[:,4].nunique() + no_positive)

	x6 = ((final_pos[5][rows[5]] ) + 1) / (train_data.iloc[:,5].nunique() + no_positive)

	#calculate the probability for for class positive
	pos_prob = x1*x2*x3*x4*x5*x6*prob_positive

	# calculate probability using m-estimate equation for class negative. where 1 is added directly by taking in account Laplace smoothing. the same thing is done for all the
	# features of dataset
	y1 = ((final_neg[0][rows[0]] ) + 1) / (train_data.iloc[:,0].nunique() + no_negative)

	y2 = ((final_neg[1][rows[1]] ) + 1) / (train_data.iloc[:,1].nunique() + no_negative)

	y3 = ((final_neg[2][rows[2]] ) + 1) / (train_data.iloc[:,2].nunique() + no_negative)

	y4 = ((final_neg[3][rows[3]] ) + 1) / (train_data.iloc[:,3].nunique() + no_negative)

	y5 = ((final_neg[4][rows[4]] ) + 1) / (train_data.iloc[:,4].nunique() + no_negative)

	y6 = ((final_neg[5][rows[5]] ) + 1) / (train_data.iloc[:,5].nunique() + no_negative)

	#calculate the probability for for class negative
	neg_prob = y1*y2*y3*y4*y5*y6*prob_negative

	#comparing the result of both classes' probability
	if pos_prob>neg_prob:
		pred_val.append('positive')
	else:
		pred_val.append('negative')

	if pred_val[index] != test_class[index]:
		error += 1

# confusion matrix is calculated and value of specificity and sensitivity is taken.
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
for i in range(len(pred_val)):
	if pred_val[i] == 'positive' and pred_val[i] == test_class[i]:
		true_positive += 1

	elif pred_val[i] == 'negative' and pred_val[i] == test_class[i]:
		true_negative += 1

	elif pred_val[i] == 'positive' and pred_val[i] != test_class[i]:
		false_positive += 1	

	elif pred_val[i] == 'negative' and pred_val[i] != test_class[i]:
		false_negative += 1	

print("Accuracy:  ",  (1-(error/len(test_class))) * 100 )
print("Specificity: ", (true_negative/ (true_negative + false_positive)))
print("Sensitivity: ", (true_positive/ (true_positive + false_negative)))

# write prediction value in predictions.csv file
file = open('test.csv')
r = csv.reader(file)
row0 = next(r)
row0.append('prediction')

count = 0
final_file = []
for item in r:
	item.append(pred_val[count])
	count+=1
	final_file.append(item)

with open('predictions.csv', 'w',newline='') as myfile:
	wr = csv.writer(myfile)
	wr.writerow(['x1','x2','x3','x4','x5','x6','target','prediction'])
	wr.writerows(final_file)