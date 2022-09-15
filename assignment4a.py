#load in packages
import pandas as pd
from tableone import TableOne, load_dataset

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