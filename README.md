# ML Quick Look up 
This repository is a quick lookup for ML Coding

## Loading Data
Advanced Loading of CSV type data
```
import pandas as pd

data = pd.read_csv(
    "data/files/complex_data_example.tsv",      # relative python path to subdirectory
    sep='\t'           # Tab-separated value file.
    quotechar="'",        # single quote allowed as quote character
    dtype={"salary": int},             # Parse the salary column as an integer 
    usecols=['name', 'birth_date', 'salary'].   # Only load the three columns specified.
    parse_dates=['birth_date'],     # Intepret the birth_date column as a date
    skiprows=10,         # Skip the first 10 rows of the file
    na_values=['.', '??']       # Take any '.' or '??' values as NA
    index_col=False,   # Row names false
    header = True      # header
)
```

## Data Properties
View head or tail of data
```
df.head()
df.tail()
```
Check the datatypes in columns 
```
df.dtypes
```
Plots of data
```
df.boxplot()
df.hist()
```
Unique Values
```
data[name].unique()
```

## Data Preprocessing
Replacing Missing Values
```
from sklearn.impute import SimpleImputer # used for handling missing data

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') imputer = imputer.fit(X[:, 1:])
X[:] = imputer.transform(X[:])
```

Delete missing values
```
df.dropna(inplace=True)
```

Encode Categorical Data
```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
```
```
cat_df_flights_lc['carrier'] = cat_df_flights_lc['carrier'].cat.codes
```

One Hot Encode Categorical Data
```
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
```

Splitting Data
```
from sklearn.model_selection import train_test_split # used for splitting training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=y)
```

Scaling 
```
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
df = sc_X.fit_transform(df)
```

PCA
```
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

pca = PCA().fit(data_rescaled)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

df=pd.DataFrame()
for i in range(pca.explained_variance_ratio_.shape[0]):
    df["pc%i" % (i+1)] = data[:,i]
df.head()
```

## ML Algos along with Hyperparameter Tuning

Logistic Regression
```
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression(multi_class='multinomial')

# Create regularization penalty space
penalty = ['l1', 'l2']]
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)

clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

best_model = clf.fit(X, y)

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

pred = best_model.predict(test)
accuracy_score(test_y,pred)
```

Decision Tree Classifier
```
from sklearn.tree import DecisionTreeClassifier
clf = clf.fit(X, Y)


```

Accuracy Score
```
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

accuracy_score(test_y,pred)
p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average !=/='binary')

```


GridSearchCV
```
from sklearn.model_selection import GridSearchCV
```
