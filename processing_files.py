"""
Created on Thu Mar  5 00:48:03 2020

@author: eyadk
"""
import os
import re
import string
import codecs
from collections import defaultdict
import nltk, pandas
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

documents_dic = defaultdict(list)

################ Reading HTML Files from all Directories ##################
count = 0
dir_path = "./doc"
directory = os.fsencode(dir_path)

for document in os.listdir(directory): # Traversing through Documents Folders
    doc_path = os.path.join(directory, document) # Joining directory path with the document name
    if count !=30:
        count = count + 1
        if os.path.isdir(doc_path): # continue if the path is a valid folder name
            pages_path = os.listdir(doc_path)
            print("######################################")
            for html_file in pages_path: # Traversing through the list of html files inside the document
                file_path = os.path.join(doc_path, html_file) # Joining doucment path with the html file
                print(file_path)
                page = codecs.open(file_path, 'r') # Open the html file
                soup = BeautifulSoup(page.read(), 'lxml')
                ocr_page = soup.find_all('div',class_='ocr_page')
                if ocr_page !=[]:
                    ocr_block = ocr_page[0].find_all('div',class_='ocrx_block')
                    for block in ocr_block:
                        ocr_par = block.find_all('p',class_='ocr_par')
                        if ocr_par !=[]:
                            for paragraph in ocr_par:
                                ocr_line = paragraph.find_all('span',class_='ocr_line')
                                if ocr_line !=[]:
                                    for line in ocr_line:
                                        ocr_info = line.find_all('span',class_='ocr_cinfo')
                                        sentence = ""
                                        for word in ocr_info:
                                            sentence = sentence + word.get_text() + " "
                                            #token = tokenizer_stemmer(word.get_text())
                                            #if token != []:
                                               # documents_dic[doc_path].append(token[0])
                                        documents_dic[doc_path].append(sentence)
                                else:
                                    print("Line element not found")
                                    continue;
                        else:
                            print("paragraph element not found")
                else:
                    print("page element not found")
    else:
        break

################ Pickeling Processed Files ##################
import pickle
f = open("document_dictionary.pkl","wb")
pickle.dump(documents_dic,f)
f.close()


total_vocab = []
total = 0
for key, value in documents_dic.items():
    total_vocab = total_vocab + value
    total = total + len(value)

print(total)

f = open("total_vocab.pkl","wb")
pickle.dump(total_vocab,f)
f.close()
