import opennre

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
model = opennre.get_model('wiki80_bertentity_softmax')
final_pars = []
#
for sent in pars:
    # print(sent)
    ents = run_re(sent)
    #print(ents)
    print(f'Entities recognized in the sentence: {len(ents)}')
    #
    if len(ents) > 1 and ents != 'Empty':
    #     ents = list(set(ents))
    #     ent_combos = result_list = list(map(tuple, itertools.combinations(
    #         ents.keys(), 2)))
        ent_combos = entities_combos(ents)
        for combo in ent_combos:
            #print(combo)
            prediction = model.infer({'text':sent, 'h':{'pos':combo[0]}, 't': {'pos': combo[1]}})
            # print(prediction)
            if prediction[0]=='subsidiary' and prediction[1]>0.85:
                print(sent)
                print(prediction)
                head_ent = ents[combo[0]]
                tail_ent = ents[combo[1]]
                print(f'{tail_ent} is a subsidiary company of {head_ent}')
                final_pars.append(sent)
            elif prediction[0]=='owned by' and prediction[1]>0.85:
                head_ent = ents[combo[0]]
                tail_ent = ents[combo[1]]
                print(sent)
                print(prediction)
                final_pars.append(sent)
                print(f'{head_ent} is a subsidiary company of {tail_ent}')

            else:
                pass
                #print('No subsidiary relationship was recognized')
    else:
        print('No relevant entities were recognized')


df = pd.DataFrame(final_pars)
df.to_csv('pars_comb_BOW_BERT.csv')