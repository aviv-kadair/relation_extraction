###This model is designed to work as a standalone. It recieves a url, extracts the text as a two-sentence
###paragraph and then sends the paragraphs to the BERT model for evaluation

from utility_functions import run_re, NER_URL, get_html, html2text, entities_combos, two_sent_maker
import opennre

import pandas as pd
import nltk
nltk.download('punkt')
import itertools

URL = input('Enter a URL: ')
html = get_html(URL)
text = html2text(html)
# split_text = text.split('.')
sent_text = nltk.sent_tokenize(text)
two_sents = two_sent_maker(sent_text)

# filter_object = [ele for ele in split_text if ele.strip()]
model = opennre.get_model('wiki80_bertentity_softmax')
pars = []
#
for sent in two_sents:
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
        if len(ent_combos)>10:
            ent_combos = ent_combos[0:11] #When a lot of entities are recognized, the combo is infinite and thus was shorten
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
                pars.append(sent)
            elif prediction[0]=='owned by' and prediction[1]>0.85:
                head_ent = ents[combo[0]]
                tail_ent = ents[combo[1]]
                print(sent)
                print(prediction)
                pars.append(sent)
                print(f'{head_ent} is a subsidiary company of {tail_ent}')

            else:
                pass

                #print('No subsidiary relationship was recognized')
    else:
        print('No relevant entities were recognized')

df = pd.DataFrame(pars)
df.to_csv('pars_BERT.csv')