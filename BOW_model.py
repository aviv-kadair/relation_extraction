from utility_functions import *
import pandas as pd

text = input('Please enter URL of choice: ')
html = get_html(text)
text = html2text(html)
reg_model = open('paragraph_reg_model.pkl','rb')
reg_model = pickle.load(reg_model)
vectorizer = open('paragraph_vectorizer.pkl','rb')
vectorizer = pickle.load(vectorizer)
pars = get_prediction(text, vectorizer, reg_model)
print(len(pars))
df = pd.DataFrame(pars)
df.to_csv('pars_BOW.csv')