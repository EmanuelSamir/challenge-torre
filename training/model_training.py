import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import scipy
import pickle5 as pickle
import dask.dataframe as dd

def clear_list(lst_of_strings):
    sl_clean = []
    for d in lst_of_strings:
        # Remove Unicode
        element_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
        # Remove Mentions
        element_test = re.sub(r'@\w+', '', element_test)
        # Lowercase the document
        element_test = element_test.lower()
        # Remove punctuations
        element_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', element_test)
        # Lowercase the numbers
        element_test = re.sub(r'[0-9]', '', element_test)
        # Remove the doubled space
        element_test = re.sub(r'\s{2,}', ' ', element_test)
        sl_clean.append(element_test)

    return sl_clean


def main():
    filename = '../data/reed_uk.csv'
    dtypes = {'category': str, 
            'city': str,
            'company_name': str,
            'geo': str,
            'job_board': str,
            'job_description': str, 
            'job_requirements': str, 
            'job_title': str, 
            'job_type': str,
            'post_date': str,
            'salary_offered': str,
            'state': str
            }

    df = pd.read_csv(filename, dtype = dtypes, )

    # Instantiate a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Convert to string column values
    s = df['job_title'].values.astype(str)
    sl = s.tolist()
    
    # Clean list
    sl_clean = clear_list(sl)

    # Training model
    X = vectorizer.fit_transform(sl_clean)

    # Convert the X as transposed matrix
    X = X.T.toarray()   # Create a DataFrame and set the vocabulary as the index
    df_lookup = pd.DataFrame(X, index=vectorizer.get_feature_names())

    # Saving model and lookup-table
    filename = '../models/search_model.sav'
    pickle.dump(vectorizer, open(filename, 'wb'))

    filename = '../models/lookup-*.h5'
    #df_lookup.to_hdf(filename, key='stage', mode='w')
    dd_lookup = dd.from_pandas(df_lookup, npartitions=2)
    dd_lookup.to_hdf(filename,'/data')

if __name__ == "__main__":
    main()