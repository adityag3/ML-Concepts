# ML Quick Look up 
This repository is a quick lookup for ML Coding

## Loading Data
```
data = pd.read_csv(
    "data/files/complex_data_example.tsv",      # relative python path to subdirectory
    sep='\t'           # Tab-separated value file.
    quotechar="'",        # single quote allowed as quote character
    dtype={"salary": int},             # Parse the salary column as an integer 
    usecols=['name', 'birth_date', 'salary'].   # Only load the three columns specified.
    parse_dates=['birth_date'],     # Intepret the birth_date column as a date
    skiprows=10,         # Skip the first 10 rows of the file
    na_values=['.', '??']       # Take any '.' or '??' values as NA
)
```
