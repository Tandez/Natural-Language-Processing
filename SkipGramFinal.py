from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from numpy import random
from scipy.special import expit
from sklearn.preprocessing import normalize
import string
from six import iteritems
import pickle

__authors__ = ['Tandez Sarkaria','Carlo Dalla Quercia','Louis-Hadrien Pion', 'Julien Crabié']
__emails__  = ['fatherchristmoas@northpole.dk','toothfairy@blackforest.no','easterbunny@greenfield.de']

path = "/Users/tandezsarkaria/Desktop/CurrentClasses/NLP/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100"     
path2 = "/Users/tandezsarkaria/Desktop/CurrentClasses/NLP/EN-SIMLEX-999.txt"
path3 = "/Users/tandezsarkaria/Desktop/CurrentClasses/NLP/Embeddings"


###### tokenization###########################

def text2sentences(path):
    sentences = []
    punctuation = set(string.punctuation)
    with open(path, encoding = "utf-8") as f:
        for l in f:
            sentences.append( l.lower().split() ) #Tokenization
        for i in range(len(sentences)):
            #We remove the punctuation:
            sentences[i] = [w for w in sentences[i] if w not in punctuation] 
            #We remove the numeric values:
            sentences[i] = [w for w in sentences[i] if w.isalpha()] 
    return sentences


sentences=text2sentences(path)

###### load test set #####################################

def loadPairs(path):
    
    data = pd.read_csv(path2,delimiter='\t',header = None) 
    data.columns = ['word1','word2','similarity']
    pairs = zip(data['word1'],data['word2'],data['similarity'])

    return pairs

pairs2 = loadPairs(path2)

class SkipGram:

    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 2, minCount = 5):
        self.sentences = sentences
        self.winSize = winSize 
        
        #The number of component in each word vector
        self.nEmbed = nEmbed 
        
        #The number of negative samples
        self.negativeRate = negativeRate 
        
        #The minimum number of occurence in the initial dataset that a word 
        #must have to be used in the model
        self.minCount = minCount 
        
        #The learning rate
        self.lr = 0.025
        
        ##### Define vocabulary and count the number of occurence of 
        # each word in the document #####
        self.vocabulary = []
        self.count={}
        for sentence in self.sentences:
            for word in sentence:
                if word not in  self.vocabulary:
                    self.vocabulary.append(word)
                    self.count[word]=1
                else:
                    self.count[word]+=1
        
        self.dictionary = {w: index for (index,w) in enumerate(self.vocabulary)}
        
        self.dictionary_new = {} #New dictionary with words that appear more than 5 times
        self.count_new = {}
        self.indices = []
        
        #Remove all the words that appear less than 5 times in the document
        i=0
        for word,occ in self.count.items():
            if int(occ) >= self.minCount:
                self.dictionary_new[word]=i
                self.indices.append(i)
                self.count_new[word]=occ
                i+=1
        
        
        ##### Computation of the unigram distribution #####
        power=0.75 #We raise to the power 3/4 as it has been shown to have better results
        self.n_voc = len(self.count_new)        
        self.unigram=np.zeros(self.n_voc) #Initialization of unigram
        
        #We define the denominator 
        total_power=sum([self.count_new[w]**power for w in self.count_new.keys()])

        #We apply the formula
        for word,occ in self.count_new.items():
            self.unigram[self.dictionary_new[word]]=self.count_new[word]**power/total_power
         




        self.window_size = winSize
        self.idx_pairs = []


        # for each word, threated as center word
        for center_word_pos in range(len(self.indices)):
        # for each window position
            for w in range(-self.window_size, self.window_size + 1):
                context_word_pos = center_word_pos + w
            # make sure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(self.indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = self.indices[context_word_pos]
                self.idx_pairs.append((self.indices[center_word_pos], context_word_idx))

                    
        self.idx_pairs = np.array(self.idx_pairs)
        
        
        
        self.weight_matrix = np.random.normal(0, 1, (self.nEmbed, self.n_voc))
        self.weight_matrix2 = np.random.normal(0, 1, size = (self.n_voc, self.nEmbed))
    

    def input_layer(self, word_idx):
        x = np.zeros(self.n_voc)
        x[word_idx] = 1.0
        return x     
    
    ##### Sigmoid function #####
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def architecture(self, idx):
        one_hot = self.input_layer(idx) 
        self.hidden_layer = np.matmul(self.weight_matrix, one_hot)
        self.output_layer = np.matmul(self.weight_matrix2,self.hidden_layer)       
        return self.hidden_layer, self.output_layer
        
    
    
    
    def train(self, epochs = 2):
        for e in range(epochs):
            for center, context in self.idx_pairs: 
                samples = []
                delta1 = np.zeros((self.nEmbed,))
                (hidden_layer, output_layer) = self.architecture(center)
                samples.append((context, 1))
                
                
                ## Selecting 5 negative samples
                Nidx = (np.random.choice(len(self.unigram), 5, p = self.unigram))
                for idx in Nidx:
                    samples.append((idx,0))          
                if context == 2:
                    print(len(samples))
                #Backpropagation
                for i, label in samples:
                     x = np.dot(self.input_layer(i).T, output_layer)
                     x = self.sigmoid(x) 
                     
                     delta = (label - x) * self.lr
                     delta1 += delta * self.weight_matrix2[i,:]
                #word_index is index of all the words being used , 1 + k negative samples
                     self.weight_matrix2[i,:] += delta * hidden_layer
                self.weight_matrix[:,center] += delta1
                    



    def save(self, path):
        #Saving the embedding matrix and the words in descending order
        Embeddings = {}
        for word,index in self.dictionary_new.items():
            Embeddings[word] = self.weight_matrix[:,index]
        
        with open(path, 'wb') as file:
            pickle.dump(Embeddings, file)    
            
    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        if(word1 in self.dictionary_new) and (word2 in self.dictionary_new):
            index1 = self.dictionary_new[word1]
            index2 = self.dictionary_new[word2]
        ##### We compute the cosine similarity #####
            dot_product = np.dot(self.weight_matrix[:,index1].T,self.weight_matrix[:,index2])
            return (dot_product + 1)/2
        else:
            return 'NA'        
        
    @staticmethod
    def load(path):
        sg = SkipGram(sentences = 'abc')
        with open(path, 'rb') as file:
            sg.weight_matrix = pickle.load(file)
        return sg
                         
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help= 'path containing training data', required=True)
    parser.add_argument('--text2', help= 'path containing test data', required=True)
    parser.add_argument('--model', help= 'path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(2)
        sg.save(opts.model)
        print('done')

    else:
        print('testing')
        pairs = loadPairs(opts.text2)
        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))

