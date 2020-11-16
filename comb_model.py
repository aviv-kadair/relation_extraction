"""This is a combined version of the B-O-W model and the openNRE bert model.
It accepts a URL and returns a list of subsidiary relationships detected.
Author: Aviv Kadair"""
import opennre
from utility_functions import *
import pandas as pd


def main():
    text = input('Please enter URL of choice: ')
    html = get_html(text)
    text = html2text(html)
    reg_model = open('paragraph_reg_model.pkl', 'rb')
    reg_model = pickle.load(reg_model)
    vectorizer = open('paragraph_vectorizer.pkl', 'rb')
    vectorizer = pickle.load(vectorizer)
    pars = get_prediction(text, vectorizer, reg_model)
    model = opennre.get_model('wiki80_bertentity_softmax')
    final_pars = []
    for sent in pars:
        ents = run_re(sent)
        if len(ents) > 1 and ents != 'Empty':
            ent_combos = entities_combos(ents)
            if len(ent_combos)>10:
                ent_combos = ent_combos[0:11] #When a lot of entities are recognized, the combo is infinite and thus was shorten
            for combo in ent_combos:
                prediction = model.infer({'text': sent, 'h': {'pos': combo[0]}, 't': {'pos': combo[1]}})
                if prediction[0] == 'subsidiary' and prediction[1] > 0.85:
                    head_ent = ents[combo[0]]
                    tail_ent = ents[combo[1]]
                    print(f'{tail_ent} is a subsidiary company of {head_ent}')
                    final_pars.append(sent)
                elif prediction[0] == 'owned by' and prediction[1] > 0.85:
                    head_ent = ents[combo[0]]
                    tail_ent = ents[combo[1]]
                    final_pars.append(sent)
                    print(f'{head_ent} is a subsidiary company of {tail_ent}')
                else:
                    pass
        else:
            print('No relevant entities were recognized')

    df = pd.DataFrame(final_pars)
    df.to_csv('pars_comb_BOW_BERT.csv')
    return final_pars



if __name__ == '__main__':
    main()
