# Imports
import streamlit as st
import os
from fastbook import *
from fastai.callback.fp16 import *
from fastai.vision.all import *
from PIL import Image
import pathlib
import os
import pandas as pd
import numpy as np
from datasets import Dataset,DatasetDict
from transformers import DataCollatorWithPadding

# For loading the model
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# Page styling
st.set_page_config(page_title="NLP", page_icon="📜", layout="wide")

# Containers for styling
header = st.container()
EDA = st.container()
images = st.container()
Model = st.container()
matrix = st.container()
with header:
    st.title("NLP: Toxic Comment Classification")
    st.write("We made a model that can classify toxic comments.")
    st.write("We will start by showing the EDA that we did.")

with EDA:
    st.header("Performing exploratory data analysis")
    col2, col3, col4 = st.columns(3)

    with st.expander("Cleaning the dataset", expanded=False):
        st.write('These are the first 7 lines in the dataset:')
        data = pd.read_csv("./train.csv")
        st.write(data.head(7))
        st.write("As you can see there are multiple columns that specify if it's a form of toxic or not. We decided that we will make one column which will specify if a comment is toxic or not. And no longer want to have multiple labels for that. We also dropped the id column since we don't really need that.")
        df_toxic = data.copy()
        df_toxic['is_toxic'] = df_toxic[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].apply(lambda row: any(row), axis=1).astype(int)
        df_toxic = df_toxic.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1)
        df_toxic = df_toxic.drop('id', axis=1)
        st.write(df_toxic.head(7))

    with st.expander("Distribution of toxic and not toxic comments", expanded=False):
        st.write(df_toxic['is_toxic'].value_counts())
        st.write("As we can see the dataset is heavily imbalenced. Let's make sure that this is fixed by reducing the number to 2000.")
        df_toxic_balanced = pd.concat([
            df_toxic[df_toxic['is_toxic'] == 0].sample(2000, random_state=42),
            df_toxic[df_toxic['is_toxic'] == 1].sample(2000, random_state=42)
        ])

        df_toxic_balanced = df_toxic_balanced.sample(frac=1, random_state=42)

        st.write(df_toxic_balanced['is_toxic'].value_counts())   
    
    with st.expander("Renaming the label", expanded=False):
        st.write("We need to rename 'is_toxic' to 'label' for it to work with the model we want to use.")
        df_toxic_balanced = df_toxic_balanced.rename(columns={'is_toxic': 'label'})
        df_toxic_balanced['label'] = df_toxic_balanced['label'].astype(float)   
        st.write(df_toxic_balanced['label'].value_counts())  


# User enters a comment
# with col2:
comment = st.text_area('Enter your comment here:')

classify_button = st.button("Classify")
if classify_button:
    st.write(comment)
    with st.spinner('Classifying...'):
        st.write('Prediction:')

with Model:
    col9, col10 = st.columns(2)
    col5, col6, col7, col8 = st.columns(4)
    
    with col9:
        st.header("Classify your comment")
        # Load the model
        # model_path = Path('./comments_model')
        # model = load_learner(model_path)
        model = torch.load('comments_model_v2.pkl', map_location=torch.device('cpu'))
        #class_names = model.dls.vocab
        

# Footer
footer="""<style>

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}


</style>
<div class="footer">
<p>Ingrid Hansen, Dina Boshnaq and Iris Loret - Big Data Group 6</p>
</div>


"""
st.markdown(footer,unsafe_allow_html=True)
    