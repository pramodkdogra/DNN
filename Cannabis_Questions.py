# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:36:15 2018

@author: prdogra
"""

# Building a ChatBot with Deep NLP
 
 
 
# Importing the libraries
import numpy as np
import tensorflow as tf
import re
#import time


from pathlib import Path
import json 
import pandas as pd 

#from __future__ import print_function


#import os
#import re
import nltk

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Bidirectional
from keras.models import Model, load_model


tf.VERSION

INPUT_LENGTH = 80
OUTPUT_LENGTH = 300
import os
import pickle

def pickle_it(path_,input_,desc):
    # Dump pickled tokenizer
    out = open(path_+ desc+".pickle","wb")
    pickle.dump(input_, out)
    out.close()   

 
########## PART 1 - DATA PREPROCESSING ##########
 

data_folder = Path("C:/Users/prdogra/OneDrive - George Weston Limited-6469347-MTCAD/prdogra/Documents/Sync/MSC_Cognitive/Year2018_2019/COS524_Natural Language Processing/Project_code/data")

# Make sure the vocabulary.txt file and the encoded datasets
v_file= data_folder / "Cannabis_faq.json"

if v_file:
    with open(v_file, 'r') as f:
        qa_dict = json.load(f)

"""        
for qa in qa_dict:
    print(qa['Questionmain'])
    print(qa['Questionalt1'])
    print(qa['Questionalt2'])
    print(qa['Questionalt3'])
    print(qa['Questionalt4'])
    print(qa['Questionalt5'])
    print(qa['Questionalt6'])
    print(qa['Questionalt7'])
    print(qa['Questionalt8'])

df_qa = pd.DataFrame(qa_dict)
#df_qa = pd.read_json(v_file, orient='columns')
print(df_qa['Questionalt1'])
   
# Getting separately the questions and the answers
questions_cannabis = []
answers_cannabis = []

# creat a temp dataframe and remove the columns answer and Qno
temp_qa=df_qa.drop(['answer','Qno'],axis=1)

### the following code reads the dataframe and flattens it to list , I did this  at thrr row level instead of column level
#to ensure all questions in the same row are rogether.

for row in temp_qa.iterrows():
    index, data = row
    for y in data.tolist(): 
        print(y)
        if str(y)!="nan":
            questions_cannabis.append(y)

## create a list to hold answers
answers_cannabis = df_qa["answer"].tolist()

"""


df_qa = pd.DataFrame(columns=['Qno','Question','Answer'])
for dict_entity in qa_dict:
    for key,val in dict_entity.items():
        print("******************************")
        qno=str.strip(str(dict_entity['Qno']))
        q_val= str.strip(str(val))
        ans=str.strip(str(dict_entity['answer']))
        if qno !=q_val:
            qno=str.strip(str(dict_entity['Qno']))
            q_val= str.strip(str(val))
            ans=str.strip(str(dict_entity['answer']))
            print(qno, q_val, ans)  
            if q_val != ans:
                df_qa = df_qa.append({'Qno':qno,'Question':q_val,'Answer':ans}, ignore_index=True)
        print("******************************")
       
questions_cannabis=df_qa['Question'].tolist()
answers_cannabis=df_qa['Answer'].tolist()

# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = text.lstrip()
    text = text.rstrip()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text
 
# Cleaning the questions
clean_questions_cannabis = []
for question in questions_cannabis:
    clean_questions_cannabis.append(clean_text(question))
 
# Cleaning the answers
clean_answers_cannabis = []
for answer in answers_cannabis:
    clean_answers_cannabis.append(clean_text(answer))
    
    
lengths = []
# lengths.append([len(nltk.word_tokenize(sent)) for sent in clean_questions]) #nltk approach
for question in clean_questions_cannabis:
    lengths.append(len(question.split()))
for answer in clean_answers_cannabis:
    lengths.append(len(answer.split()))
# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])
print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))

# Remove questions and answers that are shorter than 1 word and longer than 20 words.
min_line_length = 2
max_line_length = 10000

# Filter out the questions that are too short/long
short_questions_temp = []
short_answers_temp = []

for i, question in enumerate(clean_questions_cannabis):
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers_cannabis[i])

# Filter out the answers that are too short/long
short_questions_cannabis= []
short_answers_cannabis = []

for i, answer in enumerate(short_answers_temp):
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers_cannabis.append(answer)
        short_questions_cannabis.append(short_questions_temp[i])
        
print(len(short_questions_cannabis))
print(len(short_answers_cannabis))
 
r = np.random.randint(1,len(short_questions_cannabis))

for i in range(r, r+3):
    print(short_questions_cannabis[i])
    print(short_answers_cannabis[i])
    print()

"""

#choosing number of samples
num_samples = 30000  # Number of samples to train on.
short_questions = short_questions[:num_samples]
short_answers = short_answers[:num_samples]
#tokenizing the qns and answers
"""
short_questions_tok_cannabis = [nltk.word_tokenize(sent) for sent in short_questions_cannabis]
short_answers_tok_cannabis = [nltk.word_tokenize(sent) for sent in short_answers_cannabis]


#train-validation split
data_size_cannabis = len(short_questions_tok_cannabis)

# We will use the first 0-80th %-tile (80%) of data for the training
training_input_cannabis  = short_questions_tok_cannabis[:round(data_size_cannabis*(95/100))]
training_input_cannabis  = [tr_input[::-1] for tr_input in training_input_cannabis] #reverseing input seq for better performance
training_output_cannabis = short_answers_tok_cannabis[:round(data_size_cannabis*(95/100))]

# We will use the remaining for validation
validation_input_cannabis = short_questions_tok_cannabis[round(data_size_cannabis*(95/100)):]
validation_input_cannabis  = [val_input[::-1] for val_input in validation_input_cannabis] #reverseing input seq for better performance
validation_output_cannabis = short_answers_tok_cannabis[round(data_size_cannabis*(95/100)):]

print('training size', len(training_input_cannabis))
print('validation size', len(validation_input_cannabis))

# Create a dictionary for the frequency of the vocabulary
# Create 
vocab_cannabis = {}
for question in short_questions_tok_cannabis:
    for word in question:
        if word not in vocab_cannabis:
            vocab_cannabis[word] = 1
        else:
            vocab_cannabis[word] += 1

for answer in short_answers_tok_cannabis:
    for word in answer:
        if word not in vocab_cannabis:
            vocab_cannabis[word] = 1
        else:
            vocab_cannabis[word] += 1  
 
# Remove rare words from the vocabulary.
# We will aim to replace fewer than 5% of words with <UNK>
# You will see this ratio soon.
threshold = 0
count = 0
for k,v in vocab_cannabis.items():
    if v >= threshold:
        count += 1
 
#we will create dictionaries to provide a unique integer for each word.
WORD_CODE_START = 1
WORD_CODE_PADDING = 0


word_num_cannabis  = 2 #number 1 is left for WORD_CODE_START for model decoder later
encoding_cannabis = {}
decoding_cannabis = {1: 'START'}
for word, count in vocab_cannabis.items():
    if count >= threshold: #get vocabularies that appear above threshold count
        encoding_cannabis[word] = word_num_cannabis 
        decoding_cannabis[word_num_cannabis ] = word
        word_num_cannabis += 1

print("No. of vocab used:", word_num_cannabis)

#include unknown token for words not in dictionary

decoding_cannabis[len(encoding_cannabis)+2] = '<UNK>'
encoding_cannabis['<UNK>'] = len(encoding_cannabis)+2

dict_size_cannabis = word_num_cannabis+1
dict_size_cannabis

#1.3 Vectorizing dataset
def transform(encoding, data, vector_size=20):
    """
    :param encoding: encoding dict built by build_word_encoding()
    :param data: list of strings
    :param vector_size: size of each encoded vector
    """
    transformed_data = np.zeros(shape=(len(data), vector_size))
    for i in range(len(data)):
        for j in range(min(len(data[i]), vector_size)):
            try:
                transformed_data[i][j] = encoding[data[i][j]]
            except:
                transformed_data[i][j] = encoding['<UNK>']
    return transformed_data

#encoding training set
encoded_training_input_cannabis = transform(
    encoding_cannabis, training_input_cannabis, vector_size=INPUT_LENGTH)
encoded_training_output_cannabis = transform(
    encoding_cannabis, training_output_cannabis, vector_size=OUTPUT_LENGTH)

print('encoded_training_input_cannabis', encoded_training_input_cannabis.shape)
print('encoded_training_output_cannabis', encoded_training_output_cannabis.shape)

#encoding validation set
encoded_validation_input_cannabis = transform(
    encoding_cannabis, validation_input_cannabis, vector_size=INPUT_LENGTH)
encoded_validation_output_cannabis = transform(
    encoding_cannabis, validation_output_cannabis, vector_size=OUTPUT_LENGTH)

print('encoded_validation_input_cannabis', encoded_validation_input_cannabis.shape)
print('encoded_validation_output_cannabis', encoded_validation_output_cannabis.shape)


#2 Model Building
#2.1 Sequence-to-Sequence in Keras
import tensorflow as tf
tf.keras.backend.clear_session()
#INPUT_LENGTH = 30
#OUTPUT_LENGTH = 100

encoder_input_cannabis = Input(shape=(INPUT_LENGTH,))
decoder_input_cannabis = Input(shape=(OUTPUT_LENGTH,))




from keras.layers import SimpleRNN

encoder_cannabis = Embedding(dict_size_cannabis, 128, input_length=INPUT_LENGTH, mask_zero=True)(encoder_input_cannabis)
encoder_cannabis = LSTM(512, return_sequences=True, unroll=True)(encoder_cannabis)
encoder_last_cannabis = encoder_cannabis[:,-1,:]

print('encoder_cannabis', encoder_cannabis)
print('encoder_last_cannabis', encoder_last_cannabis)

decoder_cannabis = Embedding(dict_size_cannabis, 128, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input_cannabis)
decoder_cannabis = LSTM(512, return_sequences=True, unroll=True)(decoder_cannabis, initial_state=[encoder_last_cannabis, encoder_last_cannabis])

print('decoder_cannabis', decoder_cannabis)

# For the plain Sequence-to-Sequence, we produced the output from directly from decoder
# output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(decoder)

#2.2 Attention Mechanism
#Reference: Effective Approaches to Attention-based Neural Machine Translation's Global Attention with Dot-based scoring function (Section 3, 3.1) https://arxiv.org/pdf/1508.04025.pdf

from keras.layers import Activation, dot, concatenate

# Equation (7) with 'dot' score from Section 3.1 in the paper.
# Note that we reuse Softmax-activation layer instead of writing tensor calculation
attention_cannabis = dot([decoder_cannabis, encoder_cannabis], axes=[2, 2])
attention_cannabis = Activation('softmax', name='attention')(attention_cannabis)
print('attention', attention_cannabis)

context_cannabis = dot([attention_cannabis, encoder_cannabis], axes=[2,1])
print('context', context_cannabis)

decoder_combined_context_cannabis = concatenate([context_cannabis, decoder_cannabis])
print('decoder_combined_context', decoder_combined_context_cannabis)

# Has another weight + tanh layer as described in equation (5) of the paper
output_cannabis = TimeDistributed(Dense(512, activation="tanh"))(decoder_combined_context_cannabis)
output_cannabis= TimeDistributed(Dense(dict_size_cannabis, activation="softmax"))(output_cannabis)
print('output', output_cannabis)


model_cannabis = Model(inputs=[encoder_input_cannabis, decoder_input_cannabis], outputs=[output_cannabis])
model_cannabis.compile(optimizer='adam', loss='binary_crossentropy')
model_cannabis.summary()


training_encoder_input_cannabis = encoded_training_input_cannabis
training_decoder_input_cannabis = np.zeros_like(encoded_training_output_cannabis)
training_decoder_input_cannabis[:, 1:] = encoded_training_output_cannabis[:,:-1]
training_decoder_input_cannabis[:, 0] = WORD_CODE_START
training_decoder_output_cannabis = np.eye(dict_size_cannabis)[encoded_training_output_cannabis.astype('int')]

validation_encoder_input_cannabis = encoded_validation_input_cannabis
validation_decoder_input_cannabis = np.zeros_like(encoded_validation_output_cannabis)
validation_decoder_input_cannabis[:, 1:] = encoded_validation_output_cannabis[:,:-1]
validation_decoder_input_cannabis[:, 0] = WORD_CODE_START
validation_decoder_output_cannabis = np.eye(dict_size_cannabis)[encoded_validation_output_cannabis.astype('int')]


model_cannabis.fit(x=[training_encoder_input_cannabis, training_decoder_input_cannabis], y=[training_decoder_output_cannabis],
          batch_size=64, epochs=250)

model_cannabis.fit(x=[training_encoder_input_cannabis, training_decoder_input_cannabis], y=[training_decoder_output_cannabis],
          validation_data_cannabis=([validation_encoder_input_cannabis, validation_decoder_input_cannabis], [validation_decoder_output_cannabis]),
          #validation_split=0.05,
          batch_size=64, epochs=20)




def prediction(raw_input):
    clean_input = clean_text(raw_input)
    input_tok = [nltk.word_tokenize(clean_input)]
    input_tok = [input_tok[0][::-1]]  #reverseing input seq
    encoder_input_cannabis = transform(encoding_cannabis, input_tok, INPUT_LENGTH)
    decoder_input_cannabis = np.zeros(shape=(len(encoder_input_cannabis), OUTPUT_LENGTH))
    decoder_input_cannabis[:,0] = WORD_CODE_START
    for i in range(1, OUTPUT_LENGTH):
        output = model_cannabis.predict([encoder_input_cannabis, decoder_input_cannabis]).argmax(axis=2)
        decoder_input_cannabis[:,i] = output[:,i]
    return output

def decode(decoding, vector):
    """
    :param decoding: decoding dict built by word encoding
    :param vector: an encoded vector
    """
    text = ''
    for i in vector:
        if i == 0:
            break
        text += ' '
        text += decoding[i]
    return text



for i in range(14):
    seq_index = np.random.randint(1, len(short_questions_cannabis))
    output = prediction(short_questions_cannabis[seq_index])
    print ('Q:', short_questions_cannabis[seq_index])
    print ('A:', decode(decoding_cannabis, output[0]))
    
    

"""""""""""""""""""""""""""""""""""""""""""""""""""""

New Cases

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

pd_new_question=['What can you tell me about Cannabis', 'I do not know about Cannabis, what call you tell', 
                 'Are there other names for Cannabis', 'What are other names taken for Cannabis', 'What are other names given to Cannabis',
                 'What is Cannabis made up of' , 'what can you tell me about THC', 'what can you tell me about CBD', 
                 'can you tell me How is THC different from CBD','can you differentiate THC and CBD',
                 'can you tell me How is CBD different from THC','can you differentiate CBD and THC'                
                 ]

pd_new_question_cannabis = []
for question in pd_new_question:
    pd_new_question_cannabis.append(clean_text(question))
 

"""
K.clear_session()
graph = tf.get_default_graph()
"""

for i in range(8):
    seq_index = np.random.randint(1, len(pd_new_question_cannabis))
    output = prediction(pd_new_question_cannabis[seq_index])
    print ('Q:', pd_new_question_cannabis[seq_index])
    print ('A:', decode(decoding_cannabis, output[0]))





""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Save the model 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
path="C:\\Users\\prdogra\\OneDrive - George Weston Limited-6469347-MTCAD\\prdogra\\Documents\\Sync\\MSC_Cognitive\\Year2018_2019\\COS524_Natural Language Processing\\Project_code\\BotVersion7_Seq_2_Seq\\"
# serialize model to JSON
model_json = model_cannabis.to_json()
with open(path+"model_cannabis_attention.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_cannabis.save_weights(path+"model_cannabis_attention.h5")
print("Saved model to disk")

pickle_it(path,encoding_cannabis,"encoding_cannabis")
pickle_it(path,short_questions_cannabis,"short_questions_cannabis")

###################3. Model testing




##############################################################################################
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
LOAD THE MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from keras.models import model_from_json  
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Bidirectional, Activation, dot, concatenate 
from keras import backend as K
import pickle 
import tensorflow as tf
import nltk
import numpy as np
import re

# load json and create model
path="C:\\Users\\prdogra\\OneDrive - George Weston Limited-6469347-MTCAD\\prdogra\\Documents\\Sync\\MSC_Cognitive\\Year2018_2019\\COS524_Natural Language Processing\\Project_code\\BotVersion7_Seq_2_Seq\\"

json_file = open(path+'model_cannabis_attention.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(path+"model_cannabis_attention.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss='binary_crossentropy')

encoding_pickle_off = open(path+"encoding_cannabis.pickle","rb")
encoding_cannabis_ = pickle.load(encoding_pickle_off)


short_questions_cannabis_pickle_off = open(path+"encoding_cannabis.pickle","rb")
short_questions_cannabis_ = pickle.load(short_questions_cannabis_pickle_off)


INPUT_LENGTH = 80
OUTPUT_LENGTH = 300
WORD_CODE_START = 1
WORD_CODE_PADDING = 0


def clean_text(text):
    text = text.lower()
    text = text.lstrip()
    text = text.rstrip()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text



def prediction(raw_input):
    global graph
    clean_input = clean_text(raw_input)
    input_tok = [nltk.word_tokenize(clean_input)]
    input_tok = [input_tok[0][::-1]]  #reverseing input seq
    encoder_input_cannabis = transform(encoding_cannabis_, input_tok, INPUT_LENGTH)
    decoder_input_cannabis = np.zeros(shape=(len(encoder_input_cannabis), OUTPUT_LENGTH))
    decoder_input_cannabis[:,0] = WORD_CODE_START
    for i in range(1, OUTPUT_LENGTH):
        output = loaded_model.evaluate([encoder_input_cannabis, decoder_input_cannabis]).argmax(axis=2)
        decoder_input_cannabis[:,i] = output[:,i]
    return output

def decode(decoding, vector):
    """
    :param decoding: decoding dict built by word encoding
    :param vector: an encoded vector
    """
    text = ''
    for i in vector:
        if i == 0:
            break
        text += ' '
        text += decoding[i]
    return text

def transform(encoding, data, vector_size=20):
    """
    :param encoding: encoding dict built by build_word_encoding()
    :param data: list of strings
    :param vector_size: size of each encoded vector
    """
    transformed_data = np.zeros(shape=(len(data), vector_size))
    for i in range(len(data)):
        for j in range(min(len(data[i]), vector_size)):
            try:
                transformed_data[i][j] = encoding[data[i][j]]
            except:
                transformed_data[i][j] = encoding['<UNK>']
    return transformed_data

pd_new_question=['What can you tell me about Cannabis', 'I do not know about Cannabis, what call you tell', 
                 'Are there other names for Cannabis', 'What are other names taken for Cannabis', 'What are other names given to Cannabis',
                 'What is Cannabis made up of' , 'what can you tell me about THC', 'what can you tell me about CBD', 
                 'can you tell me How is THC different from CBD','can you differentiate THC and CBD',
                 'can you tell me How is CBD different from THC','can you differentiate CBD and THC'                
                 ]

pd_new_question_cannabis = []
for question in pd_new_question:
    pd_new_question_cannabis.append(clean_text(question))
 

K.clear_session()
graph = tf.get_default_graph()


for i in range(8):
    seq_index = np.random.randint(1, len(pd_new_question_cannabis))
    output = prediction(pd_new_question_cannabis[seq_index])
    print ('Q:', pd_new_question_cannabis[seq_index])
    print ('A:', decode(decoding_cannabis, output[0]))

