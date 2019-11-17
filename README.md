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
