# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 20:41:27 2020

@author: DELL
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
movies = []
with open("my_movies.csv") as f:
    movies = f.read()
    
#splitting data into separate transactions using "\n"
movies = movies.split("\n")
movies_list = []
for i in movies:
    movies_list.append(i.split(","))
all_movies_list = [i for item in movies_list for i in item]    

from collections import Counter,OrderedDict
item_frequencies = Counter(all_movies_list)

#after sorting 
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[1:11],left = list(range(1,11)),color='rgbkymc');plt.xticks(list(range(1,11),),items[1:11]);plt.xlabel("items")
plt.ylabel("Count");plt.xlabel("Items")

# Creating Data Frame for the transactions data 

movies_series  = pd.DataFrame(pd.Series(movies_list))
movies_series = movies_series.iloc[:166,:] # removing the last empty transaction
movies_series.columns = ["transactions"]


#creating a dummy columns for the each item in each transactions ... Using column names as item name
X = movies_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

frequent_itemsets = apriori(X, min_support=0.005, max_len=3,use_colnames = True)


# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(left = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False).head(10)




def to_list(i):
    return (sorted(list(i)))


maX = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


maX = maX.apply(sorted)

rules_sets = list(maX)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)















