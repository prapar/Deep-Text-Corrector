import numpy as np
import pandas as pd
import os
import re
import warnings
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Embedding, Dense, LSTM
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Softmax, RNN, Dense, Embedding, LSTM
import nltk
from tensorflow import keras 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping  
import datetime 
import nltk.translate.bleu_score as bleu
from tensorflow.keras.models import model_from_json
import pickle
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

direc = os.getcwd()+'\\'
tknizer_corr = pickle.load(open(direc+'input\save_models\\tknizer_corr.pkl','rb'))
tknizer_pert = pickle.load(open(direc+'input\save_models\\tknizer_pert.pkl','rb'))
embedding_matrix = pickle.load(open(direc+'input\\save_models\\embedding_matrix.pkl','rb'))

vocab_size_corr=len(tknizer_corr.word_index.keys())
vocab_size_pert=len(tknizer_pert.word_index.keys())

#vocab_size_corr = 38321
#vocab_size_pert = 37672

# Define constants
corr_index_word={}
for key,value in tknizer_corr.word_index.items():
  corr_index_word[value]=key
encoder_inputs_length = 20
decoder_inputs_length = 20
input_vocab_size = vocab_size_pert+1
output_vocab_size = vocab_size_corr+1
enc_units = 256
dec_units = 256
embedding_size = embedding_matrix.shape[1]

class Encoder(tf.keras.Model):
    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):
      super(Encoder, self).__init__()
      self.vocab_size = inp_vocab_size
      self.embedding_dim = embedding_size
      self.input_length = input_length
      self.enc_units= lstm_size 
      self.lstm_output=0
      self.lstm_state_h=0
      self.lstm_state_c=0
      self.embd_Layer = Embedding(name="encoder_embedding_layer", 
                                  input_dim=(self.vocab_size), 
                                  output_dim=self.embedding_dim, 
                                  input_length=self.input_length, 
                                  mask_zero=True)          
      self.lstm = LSTM(name="encoder_lstm",
                       units=self.enc_units, 
                       return_sequences=True, 
                       return_state=True)
      
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'inp_vocab_size': self.vocab_size,
            'embedding_size': self.embedding_dim,
            'lstm_size': self.enc_units,
            'input_length': self.input_length
        })
        return config

    def call(self,input_sequence,states):
      input_embedd = self.embd_Layer(input_sequence)        
      self.lstm_output, self.lstm_state_h, self.lstm_state_c = self.lstm(input_embedd, initial_state=states)
      return self.lstm_output, self.lstm_state_h,self.lstm_state_c
    
    def initialize_states(self,batch_size):
      self.lstm_state_h=tf.zeros((batch_size, self.enc_units))
      self.lstm_state_c= tf.zeros((batch_size, self.enc_units))
      return self.lstm_state_h, self.lstm_state_c

class Attention(tf.keras.layers.Layer):
  def __init__(self,scoring_function, att_units):
    super(Attention, self).__init__()
    self.scoring_function = scoring_function
     # Intialize variables
    if self.scoring_function=='dot':
      pass
    elif scoring_function == 'general':
      self.W1 = tf.keras.layers.Dense(att_units)
      pass
    elif scoring_function == 'concat':
      self.W1 = tf.keras.layers.Dense(att_units)
      self.W2 = tf.keras.layers.Dense(att_units)
      self.V = tf.keras.layers.Dense(1)
      pass

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scoring_function': self.scoring_function
        })
        return config

  def call(self,decoder_hidden_state,encoder_output):
    '''
    Calculate score based on the scoring_function using Bahdanu attention mechanism.
    '''
    if self.scoring_function == 'dot':
        score = tf.matmul(encoder_output, tf.expand_dims(decoder_hidden_state, 1), transpose_b=True)
        attnWeights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attnWeights * encoder_output, axis=1)
        return context_vector, attnWeights
    elif self.scoring_function == 'general':
        score = tf.matmul(encoder_output, self.W1(tf.expand_dims(decoder_hidden_state, 1)), transpose_b=True)
        attnWeights = tf.nn.softmax(score, axis=1)        
        context_vector = tf.reduce_sum(attnWeights * encoder_output, axis=1)
        return context_vector, attnWeights
    elif self.scoring_function == 'concat':
        score = self.V(tf.nn.tanh(self.W1(tf.expand_dims(decoder_hidden_state, 1)) + self.W2(encoder_output)))
        attnWeights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attnWeights * encoder_output, axis=1)        
        return context_vector, attnWeights

class One_Step_Decoder(tf.keras.Model):
  def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
      # Initialize
      super(One_Step_Decoder, self).__init__() 
      self.dense = Dense(units=tar_vocab_size)
      self.embd = Embedding(tar_vocab_size, embedding_dim, embeddings_initializer='uniform', input_length=input_length)
      self.lstm = LSTM(att_units, return_sequences=True, recurrent_initializer='glorot_uniform', return_state=True,)
      self.attention = Attention(scoring_function=score_fun, att_units=att_units)

  def call(self,input_to_decoder, encoder_output, state_h,state_c):
    dec_state_h = tf.reduce_sum([state_h, state_c], 0) # Initialise decoder hidden state
    dec_embedding_vector = self.embd(input_to_decoder)
    context_vector, attention_weights = self.attention(dec_state_h, encoder_output) #B
    concat_context_vector = tf.concat([tf.expand_dims(context_vector, 1), dec_embedding_vector], axis=2) #Concat the context vector
    decoderOut, dec_state_h, dec_state_c = self.lstm(concat_context_vector, initial_state=[state_h, state_c]) #LSTM layer
    decoder_output = tf.reshape(decoderOut, (-1, decoderOut.shape[2])) # reshape for dense layer input
    output = self.dense(decoder_output) # Dense Layer
    return output, dec_state_h, dec_state_c, attention_weights, context_vector

class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
      super(Decoder, self).__init__()
      self.one_step_decoder = One_Step_Decoder(out_vocab_size, embedding_dim, input_length, dec_units, score_fun, att_units)  

    def call(self, input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state):
      #Initialize an empty Tensor array, that will store the outputs at each and every time step
      output_tensor = tf.TensorArray(tf.float32,dynamic_size=False, size=tf.shape(input_to_decoder)[1])
      for i in range(tf.shape(input_to_decoder)[1]): #Iterate till the length of the decoder input
        decoder_input = tf.expand_dims(input_to_decoder[:,i], 1)
        # Call onestepdecoder for each token in decoder_input
        output, decoder_hidden_state, decoder_cell_state, attention_weights, context_vector = self.one_step_decoder(decoder_input, 
                                                                                                                    encoder_output,
                                                                                                                    decoder_hidden_state, 
                                                                                                                    decoder_cell_state)
        output_tensor = output_tensor.write(i, output)   # Store the output in tensorarray
      output_tensor = tf.transpose(output_tensor.stack(), [1, 0, 2])     # reshape tensor
      return output_tensor

class encoder_decoder(tf.keras.Model):
  def __init__(self):
    super().__init__()
    # create Encoder object
    self.encoder = Encoder(input_vocab_size, embedding_size, enc_units, encoder_inputs_length)
    # decoder object
    self.decoder = Decoder(output_vocab_size, embedding_size, decoder_inputs_length, dec_units, score_fun='dot', att_units=dec_units)
  
  def call(self,data):
    #Intialize encoder states, 
    input,output = data[0], data[1]  
    # Pass the encoder_sequence to the embedding layer  
    enc_state_h, enc_state_c = self.encoder.initialize_states(1024)
    enc_output, encoder_h, encoder_c = self.encoder(input, [enc_state_h, enc_state_c])
    # Decoder initial states are encoder final states, Initialize it accordingly
    # Pass the decoder sequence,encoder_output,decoder states to Decoder
    decoder_output = self.decoder(output, enc_output, encoder_h, encoder_c)    
    return decoder_output
    
def build_model():
    model = encoder_decoder()
    model.build((None,1024,20))
    # load weights into new model
    model.load_weights(direc+'input\\save_models\\model_attn.hdf5')
    return model

def predict(input_sentence):
    model = build_model()
    encoder_intial_states = model.layers[0].initialize_states(1) # Initialize encoder
    # A. Given input sentence, convert the sentence into integers using tokenizer used earlier
    tokens = pad_sequences(np.array(tknizer_pert.texts_to_sequences([input_sentence])), maxlen=20, padding='post')  
    # Pass the input_sequence to encoder
    enc_output, enc_state_h, enc_state_c = model.layers[0](tokens, encoder_intial_states)
    # Initialize index of <start> as input to decoder
    decoder_input = tf.expand_dims([tknizer_corr.word_index['<start>']] * 1, 1)
    pred = '' # Initialize empty output
    #Initialise decoder with initial states from encoder o/p 
    decoder_hidden_h = enc_state_h
    decoder_hidden_c = enc_state_c
    for i in range(20):
      predictions, decoder_hidden_h, decoder_hidden_c, attention_weights, context_vector = model.decoder.one_step_decoder(decoder_input, 
                                                                                                                        enc_output,
                                                                                                                        decoder_hidden_h, 
                                                                                                                        decoder_hidden_c)
      # get the index of the word with maximum probability of the output 
      infe_output=np.argmax(predictions,-1)    
      word_index = infe_output[0]
      # Move to next iteration for Start - exit for end.
      if word_index == 0:
        decoder_input = np.reshape(np.argmax(infe_output), (1, 1))
        continue        
      if corr_index_word[word_index] == "<end>":
        return pred
        # append predicted word to sentence
      pred=pred+' '+corr_index_word[int(word_index)]
         # updated word index with current predictions for next decoder input
      decoder_input = np.reshape(int(word_index), (1, 1)) 
    return pred