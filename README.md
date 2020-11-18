# Relation Extraction
Relation extraction is the action of extracting a relationship between two entities out of a free, unstructured text. 

This script analyses free text and detects <b>subsidiary</b> relationship between named entities. 
The script accepts a URL and extracts the free text. The text is first processed and analysed using a bag-of-words model, which outputs paragraphs expressing an acquisition
relationship. Only pararaphs with at least two entities are considered.

The extracted paragraphs are sent for further analysis in a BERT model which was trained on wiki80 dataset. 
This BERT model detects the named entities, creates pairs of entities and tests for subsidiary relationship. 
Finally, it outputs every paragraph which passes this screening test and saves it to a csv file.


# Results:
### Recall: 0.62
### Precision: 0.9

# Instructions:
Please make sure openNRE package is set and ready.<br>
Run the combined model via comb_model.py file. <br>
The B-O-W model and BERT model are also availle as a standalone models in seperate files. <br>

# Example:
<i>"The $100 million purchase helped Twitter improve mobile installation and engagement advertisements.Twitter confirmed the acquisition was made to assist in mobile marketing, 
as TapCommerce assists companies with  target advertisements  based on prior user activity." </i><br>

Output: ('Subsidiary', 0.9937208890914917) <br>
<b> TapCommerce is a subsidiary company of Twitter </b>
