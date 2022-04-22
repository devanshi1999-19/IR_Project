#!/usr/bin/env python
import requests
from readability import Document
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
#stopwords = nltk.download('stopwords')
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path
import re
import torch
import json 
from readability.readability import Document
from bs4 import BeautifulSoup
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from gensim.summarization import summarize as gensim_summarize

# using flask_restful
from flask import Response, json, Flask, jsonify, request
from flask_restful import Resource, Api

#install DL Libraries (Transformers)
!pip install transformers
!pip install bert-extractive-summarizer


from summarizer import Summarizer,TransformerSummarizer

from functools import lru_cache
#import re

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)


def read_article(text):        
  sentences =[]        
  sentences = sent_tokenize(text)    
  for sentence in sentences:        
    sentence.replace("[^a-zA-Z0-9]"," ")     
  return sentences

@lru_cache(maxsize=50)
def func(url):
  response = requests.get(url)
  doc = Document(response.text)
  html = doc.summary()
  soup = BeautifulSoup(html)
  for script in soup(["script", "style"]):
      script.decompose()
  strips = list(soup.stripped_strings)
  to_tokenize = ""
  for x in strips:
      to_tokenize += x
      to_tokenize += " "

  return summarize_tfidf(to_tokenize, 3)[0]

#LexRank
def func2(url):
    response = requests.get(url)
    doc = Document(response.text)
    html = doc.summary()
    soup = BeautifulSoup(html)
    for script in soup(["script", "style"]):
        script.decompose()
    strips = list(soup.stripped_strings)
    data2=[]
    data2.append(strips)
    lxr = LexRank(data2, stopwords=STOPWORDS['en'])
    summary = lxr.get_summary(strips, summary_size=2, threshold=.1)
    return summary
  
#T5_Algorithm
def func3(url):
  model = T5ForConditionalGeneration.from_pretrained('t5-small')
  tokenizer = T5Tokenizer.from_pretrained('t5-small')
  device = torch.device('cpu')
  response = requests.get(url)
  doc = Document(response.text)
  html = doc.summary()
  soup = BeautifulSoup(html)
  for script in soup(["script", "style"]):
    script.decompose()
  strips = list(soup.stripped_strings)
  to_tokenize = ""
  for x in strips:
      to_tokenize += x
      to_tokenize += " "
  preprocess_text = to_tokenize.strip().replace("\n","")
  t5_prepared_Text = "summarize: "+preprocess_text
  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
  # summmarizing the text 
  summary_ids = model.generate(tokenized_text,
                                      num_beams=4,
                                      no_repeat_ngram_size=2,
                                      min_length=30,
                                      max_length=100,
                                      early_stopping=True)

  output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return output

#GPT 2 summarizer
def func4(url):
  response = requests.get(url)
  doc = Document(response.text)
  html = doc.summary()
  soup = BeautifulSoup(html)
  for script in soup(["script", "style"]):
      script.decompose()
  strips = list(soup.stripped_strings)
  to_tokenize = ""
  for x in strips:
      to_tokenize += x
      to_tokenize += " "
  data=to_tokenize
  GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
  gpt_2_summary=''.join(GPT2_model(data, min_length=60))
  return gpt_2_summary
  
# BERT summarizer
def func5(url):
  response = requests.get(url)
  doc = Document(response.text)
  html = doc.summary()
  soup = BeautifulSoup(html)
  for script in soup(["script", "style"]):
      script.decompose()
  strips = list(soup.stripped_strings)
  to_tokenize = ""
  for x in strips:
      to_tokenize += x
      to_tokenize += " "
  data=to_tokenize
  data=func3(url)
  bert_model = Summarizer()
  bert_summary = ''.join(bert_model(data, min_length=60))
  return bert_summary
  
# Gensim Summarizer 
 def func6(url):
  response = requests.get(url)
  doc =  Document(response.text)
  html = doc.summary()
  soup = BeautifulSoup(html)
  for script in soup(["script", "style"]):
      script.decompose()
  strips = list(soup.stripped_strings)
  to_tokenize = ""
  for x in strips:
      to_tokenize += x
      to_tokenize += " "
  data=gensim_summarize(to_tokenize,split= True,word_count = 100)
  summary = ""
  for word in data:
    summary += word
  return summary
  
# making a resource class to get and print url
class PrintURL(Resource):

  # Corresponds to GET request
  def get(self, url):
    try:
      url = re.sub('~', '/', url)
      summary = func(url)
      whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.')
      summary = ''.join(filter(whitelist.__contains__, summary))
      #summary = re.sub('[\W_]+', '', summary)
      data = jsonify({'Summary': summary})
      data.headers.add('Access-Control-Allow-Origin', '*')
      return data

    except Exception as e:
      print(e)
      data = jsonify({'Summary': "error"})
      data.headers.add('Access-Control-Allow-Origin', '*')
      return data


# adding the defined resources along with their corresponding urls
api.add_resource(PrintURL, '/url/<string:url>')



def sentence_similarity(sent1,sent2,stopwords=None):    
  if stopwords is None:        
    stopwords = []        
  sent1 = [w.lower() for w in sent1]    
  sent2 = [w.lower() for w in sent2]
        
  all_words = list(set(sent1 + sent2))   
     
  vector1 = [0] * len(all_words)    
  vector2 = [0] * len(all_words)        
  #build the vector for the first sentence    
  for w in sent1:        
    if not w in stopwords:
      vector1[all_words.index(w)]+=1                                                             
  #build the vector for the second sentence    
  for w in sent2:        
    if not w in stopwords:            
      vector2[all_words.index(w)]+=1 
               
  return 1-cosine_distance(vector1,vector2)

def build_similarity_matrix(sentences,stop_words):
  #create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
          if idx1!=idx2:
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
    return similarity_matrix
    
def summarize_tfidf(text,top_n):
#   nltk.download('stopwords')    
#   nltk.download('punkt')
  stop_words = stopwords.words('english')    
  summarize_text = []
  # Step1: read text and tokenize    
  sentences = read_article(text)
  # Step2: generate similarity matrix            
  sentence_similarity_matrix = build_similarity_matrix(sentences,stop_words)
  # Step3: Rank sentences in similarity matrix
  sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
  scores = nx.pagerank(sentence_similarity_graph)
  # Step4: sort the rank and place top sentences
  ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
  
  # Step5: get the top n number of sentences based on rank
  for i in range(top_n):
    summarize_text.append(ranked_sentences[i][1])
  # Step6 : output the summarized version
  return " ".join(summarize_text),len(sentences)
  


# driver function
if __name__ == '__main__':

  app.run(debug = True)
