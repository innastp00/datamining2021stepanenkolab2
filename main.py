#%%

import os
import pandas as pd
import matplotlib.pyplot as plt
path = r"F:\labapython\out\dataset.csv"
df = pd.read_csv(path)
df.head()

#%%

pham = (df[df['v1'] == 'ham'].count().v1) / (df.count().v1)
pspam = (df[df['v1'] == 'spam'].count().v1) / (df.count().v1)

print(f'p(HAM) = {pham} | p(SPAM) = {pspam}')

#%%

from collections import defaultdict

dict_ham = defaultdict(int)
dict_spam = defaultdict(int)
dict_global = defaultdict(int)

with open(r"F:\labapython\out\ham.txt") as f:
    for line in f:
       (key, val) = line.split()[:2]
       dict_ham[key] = int(val)

with open(r"F:\labapython\out\spam.txt") as f:
    for line in f:
       (key, val) = line.split()[:2]
       dict_spam[key] = int(val)

for d in (dict_ham, dict_spam):
    for key, value in d.items():
        dict_global[key] += int(value)

print(len(dict_global.items()), len(dict_spam.items()), len(dict_ham.items()))

#%%

import numpy as np
def f_spamword(word):
    return int(dict_spam[word]) / len(dict_spam.keys())
def f_hamword(word):
    return int(dict_ham[word]) / len(dict_ham.keys())

def funk_ham(text):
    body = set(text.split())
    return np.prod(np.array([f_hamword(word) for word in body]))
def funk_spam(text):
    body = set(text.split())
    return np.prod(np.array([f_spamword(word) for word in body]))

def p_ham_bodytext(text):
    return pham * funk_ham(text)
def p_spam_bodytext(text):
    return pspam * funk_spam(text)

#%%

import re
import nltk
from collections import defaultdict

def process_text(text):
    text = re.sub(r'[^a-zA-Z^ \t\n\r]', '', text)
    text = text.lower()
    stop = "''a,to,the,in,have,has,had,do,does,did,am,is,are,shall,will,should,would,may,might,must,can,could,a,to,the,in,be,being,been"
    stop_words = stop.split('/')
    text = ' '.join([word for word in text.split(' ') if word not in stop_words])
    ps = nltk.stem.SnowballStemmer('english')
    text = ' '.join([ps.stem(word) for word in text.split(' ')])
    return text

def count_words(text):
    for word in text.split():
        dict_ham[word] += 1
        dict_spam[word] += 1
        dict_global[word] += 1

def predict(text):
    text = process_text(text)
    count_words(text)
    p1 = p_ham_bodytext(text)
    p2 = p_spam_bodytext(text)
    p1, p2 = (p1/(p1+p2)), (p2/(p1+p2))
    return f'Ham : {p1:.7}% \nSpam : {p2:.7}%'

print(predict(df.v2[0]))

#%%

df.head()

#%%

while 1:
    message = str(input('\nEnter message pls:'))
    if message != '0':
        print(predict(message))
    else:
        break

#%%



#%%


