# Loan-Eligibility-prediction
# importing the pakages

import pandas as pd
import numpy as npg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
     

dataset = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
dataset.head(10)
     
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25

dataset.describe() #describing our data on a high level(count and distribution)
     
Hours	Scores
count	25.000000	25.000000
mean	5.012000	51.480000
std	2.525094	25.286887
min	1.100000	17.000000
25%	2.700000	30.000000
50%	4.800000	47.000000
75%	7.400000	75.000000
max	9.200000	95.000000

dataset.isnull().sum() # will give the missing values
     
Hours     0
Scores    0
dtype: int64

# plotting the 2D graphfor our points

dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores achieved')
plt.show()
     


# slpitting our data into features (attributes) and labels (what we are predicting)

features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, 1].values
     

# dividing our full dataset into training and test dataset(80% and 20% respectively)

x_train ,x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
print(len(labels), len(y_train), len(y_test )) # checking the divide
     
25 20 5

# training our model by fitting on the training data

regressor = LinearRegression() #instantiating the model class
regressor.fit(x_train, y_train )
print('Training done')
     
Training done

#prediction on the training data

train_pred =regressor.predict(x_train)
train_pred
     
array([39.67865467, 20.84840735, 79.32128059, 70.40168976, 12.91988217,
       52.56250809, 78.33021494, 34.72332643, 84.27660883, 93.19619966,
       62.47316457, 36.70545772, 28.77693254, 56.52677068, 28.77693254,
       86.25874013, 26.79480124, 49.58931115, 90.22300272, 46.6161142 ])

# plotting the regression line

line = regressor.coef_*features+regressor.intercept_
ax = plt.gca()

plt.scatter(features ,labels)
plt.plot(features, line);
plt.show()
     


# since our model is ready, we shall fit it on test data and predict

print(x_test)
y_pred = regressor.predict(x_test)
y_pred
     
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
array([16.88414476, 33.73226078, 75.357018  , 26.79480124, 60.49103328])

# compare actual scores with predicted scores

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
     
Actual	Predicted
0	20	16.884145
1	27	33.732261
2	69	75.357018
3	30	26.794801
4	62	60.491033

# now it's time to predict score(percentage) of a student who studied for 9.25 hrs

hours = 9.25
own_pred = regressor.predict([[hours]])
print(f'number of hours studied: {hours}')
print(f'scores achieved: {own_pred}')
     
number of hours studied: 9.25
scores achieved: [93.69173249]

# calculating the performance metrics(MAE) on testing data if it's lower than the MAE of naive data(training data), the model is good

print(f'Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}')
     
Mean Absolute Error: 4.183859899002982
