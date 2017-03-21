# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:18:23 2017

@author: John
"""
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import string, pandas as pd, numpy as np
from nltk import corpus
import matplotlib.pyplot as plt

path = "C:/Users/John/Documents/Python Stuff/Text Ranker/"

def get_freq_list():
    """From http://norvig.com/ngrams/count_1w.txt"""
    freqs = pd.read_csv(path + 'freq_dist.txt', sep='\t', header = None, names = ["Word", "Freq"])
    freqs["Percent"] = freqs["Freq"] / np.sum(freqs["Freq"])
    return freqs

def get_text(text):
    with open(text, 'r') as myfile:
        test = myfile.read().replace('\n', '')
    test = corpus.gutenberg.raw('melville-moby_dick.txt')
    return test


test = get_text(path + 'test.txt')
freqs = get_freq_list()
freqs_dict = pd.Series(freqs.Percent.values,index=freqs.Word).to_dict()

#lmtzr = WordNetLemmatizer()
#porter_stemmer = PorterStemmer()
#
#words = []
#
#for word in test.split(" ")[:50]:
#    original = word
#    noun = lmtzr.lemmatize(word, 'n')
#    verb = lmtzr.lemmatize(word, 'v')
#    if noun == original and not verb == original:
#        words.append(verb)
#    elif verb == original and not noun == original:
#        words.append(noun)
#    else:
#        words.append(noun)

test = test.lower()
test = " ".join(test.split())
test = "".join(l for l in test if l not in string.punctuation)
test = test.split(" ")
test = [s for s in test if not any(c.isdigit() for c in s)]


word_counts = dict(Counter(test))
word_pcts = dict()

total = len(test)

for key, value in word_counts.items():
    word_pcts[key] = (value / total)

word_freqs = pd.DataFrame(columns=['Word', 'Frequency'])
word_freqs["Word"] = word_pcts.keys()
word_freqs["Frequency"] = word_pcts.values()

word_freqs.sort_values(by=["Frequency"], axis=0, ascending=False, inplace=True)

word_freqs["Expected"] = word_freqs['Word'].map(freqs_dict)

word_freqs = word_freqs.fillna(0)

word_freqs["Delta"] = np.subtract(word_freqs['Frequency'], word_freqs["Expected"])

print(np.sum(word_freqs["Delta"]))

expected = word_freqs["Expected"]
data = word_freqs["Frequency"].tolist()

plt.hist(data, bins = 50)
plt.title("Word Frequency Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.yscale('log', nonposy='clip')
plt.show()




