#load in packages
import pandas as pd
from tableone import TableOne, load_dataset
import numpy as np

#using sample data to test functions of tableone
data = pd.read_csv('data.csv')
list(data)
data_columns = ['Age', 'HR', 'Smoke', 'Group' ]
data_categories = ['Smoke', 'Group']
data_groupby = ['Smoke']
data_table1 = TableOne(data, columns=data_columns, 
    categorical=data_categories, groupby=data_groupby, pval=False)
data_table1 
print(data_table1.tabulate(tablefmt = "fancy_grid")) #pretty display of data in terminal
#saving pretty data into new csv
data_table1.to_csv('data/pretty_table1_data.csv')

#scipy lectures page codes
#load sample data
df = pd.read_csv('data/scipy_brain.csv')
df
#3 numpy arrays displayed in pandas dataframe
import numpy as np
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)
pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})  

#rows & columns
df.shape
#column names
df.columns
#print data in column 
print(df['Gender'])  

df[df['Gender'] == 'Female']['VIQ'].mean()

#quick view of data
pd.DataFrame.describe(df)
#groupby
groupby_gender = df.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
        print((gender, value.mean()))

groupby_gender.mean()

#scatter matrices
from pandas.tools import plotting
plotting.scatter_matrix(df[['Weight', 'Height', 'MRI_Count']])  
plotting.scatter_matrix(df[['PIQ', 'VIQ', 'FSIQ']]) 

#hypothesis testing
from scipy import stats
#1 sample t-test
stats.ttest_1samp(df['VIQ'], 0)  
#2 sample t-test
female_viq = df[df['Gender'] == 'Female']['VIQ']
male_viq = df[df['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)   

#paired tests
stats.ttest_ind(df['FSIQ'], data['PIQ'])   
stats.ttest_rel(df['FSIQ'], data['PIQ'])   
stats.ttest_1samp(df['FSIQ'] - data['PIQ'], 0)  
stats.wilcoxon(data['FSIQ'], data['PIQ'])  

#liner regression
import numpy as np
x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
# Create a data frame containing all the relevant variables
data = pandas.DataFrame({'x': x, 'y': y})
#specify an OLS model and fit it
from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()
# inspect stats from fit
print(model.summary())  

#categorical variables
data = pd.read_csv('data/scipy_brain.csv', sep=';', na_values=".")
#compare IQ of male and female using a linear model
model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary())  
#An integer column can be forced to be treated as categorical using:
model = ols('VIQ ~ C(Gender)', data).fit()


#Link to t-tests
data_fisq = pd.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pd.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pd.concat((data_fisq, data_piq))
print(data_long)  
model = ols("iq ~ type", data_long).fit()
print(model.summary())  

#multiple regression
data = pd.read_csv('data/iris.csv')
model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary())  
#f test
print(model.f_test([0, 1, -1, 0]))  

#seaborn for statistical exploration
import seaborn
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
    kind='reg')  
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                 kind='reg', hue='SEX')  
#reset default
from matplotlib import pyplot as plt
plt.rcdefaults()

#plotting univariate regression
seaborn.lmplot(y='WAGE', x='EDUCATION', data=data)  

#testing for interactions
result = sm.ols(formula='wage ~ education + gender + education * gender',
                data=data).fit()    
print(result.summary()) 

