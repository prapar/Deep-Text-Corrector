import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import re
import os
from model import *

direc = os.getcwd()+'\\'
#attn_model = model.encoder_decoder()
#Build the model
#attn_model.build((None,1024,20))
# load weights into new model
#attn_model.load_weights(direc+'input\save_models\model_attn.hdf5')

st.title('Deep Text Corrector')
st.subheader('Type in the text to be corrected')

with st.form("my-form", clear_on_submit=True):
        sentence = st.text_input('Input your sentence here:')
        submitted = st.form_submit_button("Submit")    
        
if(submitted):
    
    if(sentence==''):
        st.write('Entered text is empty!')
        
    elif(len(sentence.split())<3 or len(sentence.split())>20):
        st.write('Enter sentence with word count between 3 and 20.')
    
    if (len(sentence.split())>3 and len(sentence.split())<20):    
        sentence = re.sub(r"[^A-Za-z0-9\'\s]","", sentence)
        sentence = sentence.lower()
        st.write('Before Correction : '+sentence)
        st.write('After Correction  : '+predict(sentence))            

