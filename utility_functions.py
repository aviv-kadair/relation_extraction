import requests
import json
import pandas as pd
import pickle
from bs4 import BeautifulSoup as bs
import itertools
from funcy import chunks
import re, string
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import numpy as np


NER_URL = 'http://10.12.0.19:5000'


def entities_combos(entities_dict):
    ent_combos = list(map(tuple, itertools.combinations(entities_dict.keys(), 2)))
    return ent_combos

def two_sent_maker(sentence_list):
    two_sent_list = ["".join(chunk) for chunk in chunks(2, sentence_list)]
    return two_sent_list

def html2text(html):
    soup = bs(html, features='lxml')
    for script in soup(["script", "style"]):
        script.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return soup.get_text(separator=' ').strip()


def get_html(url):
    user_agent = {'User-agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=user_agent, timeout=10)
    html = resp.text
    return html


def run_re(text, timeout=100, NER_URL=NER_URL):
    NER_URL = NER_URL
    quer = {
        'text': text,
        'run_regex_ner': True}
    try:
        res = requests.post(NER_URL, data=json.dumps(quer), headers={'Content-Type': 'application/json'}, timeout=None)
        res_exp = res.json()
        ner_res = res_exp['ner_results']
        ents = {}
        if len(ner_res) > 1:
            for item in ner_res:
                if item['entity_type'] == 'ORG':
                    # print(item['text'])
                    ents[(item['start'], item['end'])]=item['text']
            return ents
        else:
            return 'Empty'
    except Exception as e:
        print(f'Warning: Exception {type(e)}')

###BOW exclusive functions##

def lemmatize_text(text):
  """Lemmatize and tokenize each sentence"""
  word_tokens = nltk.word_tokenize(text)
  lemmatizer  = WordNetLemmatizer()
  return [lemmatizer.lemmatize(w, pos='v') for w in word_tokens]

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def remove_nonalpha_num(text):
  text = [word for word in text if word.isalpha()]
  return text

def price_tag(text):
  text = re.sub('\n', '', text)
  _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

  text = text.strip()
  return re.sub('[\$|\€]\d+\.?\d*','PRICE_TAG', text)

def string_joiner(words_list):
  processed_sent = ' '.join(words_list)
  return processed_sent

def preprocess_pipe(text):

  cleaned_sent = price_tag(text)
  cleaned_sent = remove_punctuations(cleaned_sent)
  cleaned_sent = lemmatize_text(cleaned_sent)
  cleaned_sent = remove_nonalpha_num(cleaned_sent)
  processed_sent = string_joiner(cleaned_sent)
  return processed_sent

def string_len(processed_sent):
  return len(processed_sent)

def get_prediction(text, vectorizer, reg_model):
  pars = []
  sent_text = nltk.sent_tokenize(text)
  two_sents = two_sent_maker(sent_text)
  sents = pd.DataFrame(two_sents)
  sents.to_csv('two_sents_output.csv')
  for par in two_sents:

      processed_sent = preprocess_pipe(par)
      input=[processed_sent]
      string_len_val = string_len(processed_sent)
      if string_len_val>35:
          vector_words = vectorizer.transform(input)
          vector_words = vector_words.toarray()
          sample_sent = np.append (vector_words, [string_len_val])
          prediction = int(reg_model.predict(sample_sent.reshape(1,-1)))
          if prediction == 1:
              ents = run_re(par)
              if ents != 'Empty' and len(ents)>1:
                  print('Subsidiary relationship was recognised')
                  print(par)
                  pars.append(par)
  return pars