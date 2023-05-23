# installed quandl, pandas, sklearn packages
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
# sklearn - you want data to be between -1 and 1
#helps with accuracy

# my quandl API key: wxUmbVAeZB1eL4eptgXn
# Using "Big Mac Index - Romania"
# "The Big Mac Index is an informal measure of currency exchange rates at ppp.
# It measures their value against a similar basket of goods and services,
# in this case a Big Mac.Differing prices at market exchange rates would imply
# that one currency is under or overvalued."


#df == "dataframe"
quandl.ApiConfig.api_key = 'wxUmbVAeZB1eL4eptgXn'
df = quandl.get("ECONOMIST/BIGMAC_ROU", authtoken="wxUmbVAeZB1eL4eptgXn")
#print(df.head())

df = df[['local_price', 'dollar_ex', 'dollar_price', 
        'dollar_adj_valuation', 'euro_adj_valuation']]
#print(df.head())

#dummy ratios, no meaning to the values calculated below
df['local_to_dollar'] = (df['local_price'] / df['dollar_price'])
df['local_to_dolex'] = (df['local_price'] / df['dollar_ex'])

df = df[['local_price', 'dollar_price', 'dollar_ex', 
        'local_to_dollar', 'local_to_dolex']]
#print(df.head())

forecast_col = 'local_price'
# fillna method fills in NaN data, NaN data will 
# be treated as an outlier in the data set
df.fillna(-99999, inplace=True)

# line below effectively predicts data 10% out
forecast_out = int(math.ceil(0.01*len(df)))

#shifting columns negatively (upwards like a spreadsheet)
# each label row will be adjusted local price 10% 
# (of specified time frame) into future
df['label'] = df[forecast_col].shift(-forecast_out)
#printdf.head()
df.dropna(inplace=True)
print(df.head())



#define X (the features) and y (the labels)
X = np.array(df.drop(['label'],axis=1)) # .drop returns new array, everything but label column
y = np.array(df['label'])

X = preprocessing.scale(X) # scaling helps with training, may increase processing time
#X = X[:-forecast_out+1] # want to make sure we have x's only where we have values for y
#df = df.dropna(inplace=True) - caused error becuase operates in-place and does not return new df
y = np.array(df['label'])
#print(len(X), len(y))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
#clf = classifier
clf1 = LinearRegression()
clf1.fit(X_train, y_train)
accuracy1 = clf1.score(X_test, y_test)
print(accuracy1)

#support vector machines
clf2 = svm.SVR(kernel='poly')
clf2.fit(X_train, y_train)
accuracy2 = clf2.score(X_test, y_test)
print(accuracy2)















