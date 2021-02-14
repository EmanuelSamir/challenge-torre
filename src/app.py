import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import scipy
import pickle5 as pickle
from bokeh.plotting import figure
import streamlit as st
import matplotlib.pyplot as plt
# For pie chart

from bokeh.io import output_file, show
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum




@st.cache
def load_data():
    #filename = '../data/reed_uk_cleaner.csv'
    #df = pd.read_csv(filename)
    url = 'https://drive.google.com/file/d/1Ayq9YelRlYxFKjFHEMWlfFReF0MO5aqN/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df = pd.read_csv(path)

    # Change post_date datatype
    df['post_date'] = pd.to_datetime(df['post_date'], format='%m/%d/%Y', errors='ignore')
    return df

def load_model(df):
    # Instantiate a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Convert to string
    s = df['job_title'].values.astype(str)
    sl = s.tolist()

    sl_clean = []
    for d in sl:
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

    # Training model
    X = vectorizer.fit_transform(sl_clean)

    # Convert the X as transposed matrix
    X = X.T.toarray()# Create a DataFrame and set the vocabulary as the index
    df_tmp = pd.DataFrame(X, index=vectorizer.get_feature_names())
    return (vectorizer, df_tmp)

def pred_model(model, lookup, query):
    q = [query]
    q_vec = model.transform(q).toarray().reshape(lookup.shape[0],)
    sim = {}
    # Calculate the similarity
    for i in range(lookup.shape[0]):
        sim[i] = np.dot(lookup.loc[:, i].values, q_vec) / np.linalg.norm(lookup.loc[:, i]) * np.linalg.norm(q_vec)
    # Sort the values 
    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    # Return similarity values
    return sim_sorted




def main():
    df = load_data() 
    model, lookup = load_model(df)
    # TODO Add try except to collect data
    st.title("Data Visualization of job database")
    
    

    modes =  (
            'When each category jobs are usually posted?', 
            'How are jobs distributed per category?', 
            'Where are usually located the jobs?',
            'Let\'s find a job'
            )


    mode = st.sidebar.selectbox(
        "What do you want to see?",
        modes
    )

    st.header(mode)

    if mode == modes[0]:
        
        categories = list(df['category'].value_counts().index)
        categories_selected = st.multiselect(
        "Choose a category", categories, ["education jobs"]
        )
        datetype = st.selectbox('Which date type in?', ('months', 'days'))
        if not categories_selected:
            st.error("Please select at least one category.")
        else:
            df_tmp = df.loc[df['category'].isin(categories_selected)]
            #st.dataframe(df_tmp)
            if (datetype == 'months'):
                groups = df_tmp["post_date"].groupby(df_tmp["post_date"].dt.month).count()
                plot_title, plot_xlabel, plot_ylabel = 'Distribution per Months', 'Months', '# Posts'
            else:
                groups = df_tmp["post_date"].groupby(df_tmp["post_date"].dt.day).count()
                plot_title, plot_xlabel, plot_ylabel = 'Distribution per Days', 'Days', '# Posts'

            days = list(groups.index)#[str(x) for x in list(groups.index)]
            quantity = list(groups.values)
            #st.write(days)
            p = figure(title=plot_title, x_axis_label=plot_xlabel, y_axis_label=plot_ylabel)
            p.vbar(x=days, top=quantity, width=0.9)
            st.bokeh_chart(p)

    elif mode == modes[1]:
        counts = df['category'].value_counts()
        jobs = list(counts.index)
        num = list(counts.values)
        f, ax = plt.subplots(figsize= (14,14))
        ax.pie(num, labels = jobs)
        ax.set_title('Job Category Distribution')
        st.write(f)


    elif mode == modes[2]:
        df_tmp = df.dropna()
        st.map(df_tmp)

    elif mode == modes[3]:
        query = st.text_input('What type of jobs are you looking for', 'manager')
        sim_sorted = pred_model(model, lookup, query)
        df_tmp = df.drop(['lat', 'lon' ], axis = 1)
        df_tmp.reset_index(drop=True, inplace=True)

        if sim_sorted:
            for k, v in sim_sorted:
                if v != 0.0:
                    st.write("Similarity:", v)
                    st.table(df_tmp.iloc[[k]])
        else:
            st.error("Sorry. Not found.")




if __name__ == "__main__":
    main()