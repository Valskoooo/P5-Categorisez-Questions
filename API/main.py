'''
P5 - API - StackOverflow Tag Prediction
'''

import streamlit as st
import myfct
from gensim.models import Word2Vec

model_w2v = Word2Vec.load('word2vec.model')

st.title('StackOverflow Tag Prediction')
user_input = st.text_input('Entrer une phrase :')

token = myfct.preprocess_text(user_input)
filtered_words = myfct.filter_words_in_vocab(token, model_w2v)

# Si la liste filtrée n'est pas vide, trouver les mots les plus similaires
if filtered_words:
    similar_words = model_w2v.wv.most_similar(positive=filtered_words, topn=5)
    st.text([tuple_element[0] for tuple_element in similar_words])
else:
    st.text("Aucun des mots fournis n'est dans le vocabulaire du modèle.")