"""This B-O-W model accepts a URL and detects whether an acquisition was mentioned in any
of the paragraphs.
Author: Aviv Kadair
"""
from utility_functions import *
import pandas as pd

def main():
    text = input('Please enter URL of choice: ')
    html = get_html(text)
    text = html2text(html)
    reg_model = open('paragraph_reg_model.pkl','rb')
    reg_model = pickle.load(reg_model)
    vectorizer = open('paragraph_vectorizer.pkl','rb')
    vectorizer = pickle.load(vectorizer)
    pars = get_prediction(text, vectorizer, reg_model)
    df = pd.DataFrame(pars)
    df.to_csv('pars_BOW.csv')


if __name__ == '__main__':
    main()