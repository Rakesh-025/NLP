#Topic Modeling and text summeruzation on tweets data set
#importing the packages
import pandas as pd
import re

#Defining the terms to be cleaned from the data set
HANDLE = '@/W+'
LINK = 'https?://t\.co/\w+'
SPECIAL_CHARS = '&lt;|&lt;|&amp;|#'

#Creating a custom function to clean the data
def clean(text):
    text = re.sub(HANDLE, ' ', text)
    text = re.sub(LINK, ' ', text)
    text = re.sub(SPECIAL_CHARS, ' ', text)
    return text

# Importing the tweets data set into Python
tweets = pd.read_csv("C:/Users/kaval/OneDrive/Desktop/Assignments/NLP/Data.csv", usecols=['text'])
tweets.head(10)

#Appling clean function to the tweets data set
tweets['text'] = tweets.text.apply(clean)

#Importingpackages to do the Topic modeling
from gensim.parsing.preprocessing import preprocess_string

tweets = tweets.text.apply(preprocess_string).tolist()

from gensim import corpora
from gensim.models.ldamodel import LdaModel

dictionary = corpora.Dictionary(tweets)
corpus = [dictionary.doc2bow(text) for text in tweets]

#Selecting number of topics to apply Latent Dirichlet Allocation (LDA)
NUM_TOPICS = 3
ldamodel = LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=10)

ldamodel.print_topics(num_words=5)
#Creating coherance model to the selected topics to get the ratings
from gensim.models.coherencemodel import CoherenceModel

#Defining coherance scores
def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()

def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        ldamodel = LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=2)
        coherence = calculate_coherence_score(tweets, dictionary, ldamodel)
        yield coherence

#Applying coherence scores
min_topics, max_topics = 5,8
coherence_scores = list(get_coherence_values(min_topics, max_topics))

#Visualization of coharence score on graph
import matplotlib.pyplot as plt

x = [int(i) for i in range(min_topics, max_topics)]

ax = plt.figure(figsize=(10,8))
plt.xticks(x)
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores', fontsize=10);

#Summarization for the tweets data set
import nltk
nltk.download('stopwords')

from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest

#Adding puctuation marks into the stop words
STOPWORDS = set(stopwords.words('english') + list(punctuation))
MIN_WORD_PROP, MAX_WORD_PROP = 0.1, 0.9


#Defining the custom function to count the frequencies of the words
def compute_word_frequencies(word_sentences):
    words = [word for sentence in word_sentences 
                     for word in sentence 
                         if word not in STOPWORDS]
    counter = Counter(words)
    limit = float(max(counter.values()))
    word_frequencies = {word: freq/limit 
                                for word,freq in counter.items()}
    #Dropping the words which are very common and uncommon
    word_frequencies = {word: freq 
                            for word,freq in word_frequencies.items() 
                                if freq > MIN_WORD_PROP 
                                and freq < MAX_WORD_PROP}
    return word_frequencies

#Defing the custom function to find the score of the sentences
def sentence_score(word_sentence, word_frequencies):
    return sum([ word_frequencies.get(word,0) 
                    for word in word_sentence])

#Defining the custom function to summerize the data set
def summarize(text:str, num_sentences=3):
    """
    Summarize the text, by return the most relevant sentences
     :text the text to summarize
     :num_sentences the number of sentences to return
    """
    text = text.lower()
    
    sentences = sent_tokenize(text) 
    
    word_sentences = [word_tokenize(sentence) for sentence in sentences]
    
    word_frequencies = compute_word_frequencies(word_sentences)
    
    # Calculating the scores of the sentences
    scores = [sentence_score(word_sentence, word_frequencies) for word_sentence in word_sentences]
    sentence_scores = list(zip(sentences, scores))
    
    # Ranking the sentences
    top_sentence_scores = nlargest(num_sentences, sentence_scores, key=lambda t: t[1])
    
    # Returning the top ranked sentence
    return [t[0] for t in top_sentence_scores]

#Opening the tweets data setinto Python
with open('C:/Users/kaval/OneDrive/Desktop/Assignments/NLP/Data.csv', 'r') as file:
    tweet = file.read()
tweet

#Appling summerization function to the tweets data set
len(sent_tokenize(tweet))

summarize(tweet)
#Summerising the second sentence
summarize(tweet, num_sentences=2)

