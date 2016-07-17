#-*- coding: utf-8 -*-
#code from http://www.jianshu.com/p/7e233ef57cb6
#2016年 03月 03日 星期四 11:01:05 CST by Demobin

import numpy as np
import codecs
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential,Graph, model_from_json
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split

from src import viterbi
from src.dataUtils import sent2vec, corpus_tags, loadCwsInfo, loadCwsData

def train(cwsInfo, cwsData, modelPath, weightPath):

    (initProb, tranProb), (vocab, indexVocab) = cwsInfo
    (X, y) = cwsData

    train_X, test_X, train_y, test_y = train_test_split(X, y , train_size=0.9, random_state=1)

    #train_X = [ [ [],[],...,[] ] ]  train_y = [ [],[],...,[] ]
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    outputDims = len(corpus_tags)
    Y_train = np_utils.to_categorical(train_y, outputDims)
    Y_test = np_utils.to_categorical(test_y, outputDims)
    batchSize = 128
    vocabSize = len(vocab) + 1
    wordDims = 100
    maxlen = 7
    hiddenDims = 100

    w2vModel = Word2Vec.load('model/sougou.char.model')
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    embeddingWeights = np.zeros((vocabSize + 1, embeddingDim))
    for word, index in vocab.items():
        if word in w2vModel:
            e = w2vModel[word]
        else:
            e = embeddingUnknown
        embeddingWeights[index, :] = e

    #LSTM
    model = Sequential()
    model.add(Embedding(output_dim = embeddingDim, input_dim = vocabSize + 1,
        input_length = maxlen, mask_zero = True, weights = [embeddingWeights]))
    model.add(LSTM(output_dim = hiddenDims, return_sequences = True))
    model.add(LSTM(output_dim = hiddenDims, return_sequences = False))
    model.add(Dropout(0.5))
    model.add(Dense(outputDims))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    result = model.fit(train_X, Y_train, batch_size = batchSize,
                    nb_epoch = 20, validation_data = (test_X,Y_test), show_accuracy=True)

    j = model.to_json()
    fd = open(modelPath, 'w')
    fd.write(j)
    fd.close()

    model.save_weights(weightPath)

    return model

def loadModel(modelPath, weightPath):

    fd = open(modelPath, 'r')
    j = fd.read()
    fd.close()

    model = model_from_json(j)

    model.load_weights(weightPath)

    return model


# 根据输入得到标注推断
def cwsSent(sent, model, cwsInfo):
    (initProb, tranProb), (vocab, indexVocab) = cwsInfo
    vec = sent2vec(sent, vocab, ctxWindows = 7)
    vec = np.array(vec)
    probs = model.predict_proba(vec)
    #classes = model.predict_classes(vec)

    prob, path = viterbi.viterbi(vec, corpus_tags, initProb, tranProb, probs.transpose())

    ss = ''
    for i, t in enumerate(path):
        ss += '%s/%s '%(sent[i], corpus_tags[t])
    ss = ''
    word = ''
    for i, t in enumerate(path):
        if corpus_tags[t] == 'S':
            ss += sent[i] + ' '
            word = ''
        elif corpus_tags[t] == 'B':
            word += sent[i]
        elif corpus_tags[t] == 'E':
            word += sent[i]
            ss += word + ' '
            word = ''
        elif corpus_tags[t] == 'M':
            word += sent[i]

    return ss

def cwsFile(fname, dstname, model, cwsInfo):
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()

    fd = open(dstname, 'w')
    for line in lines:
        rst = cwsSent(line.strip(), model, cwsInfo)
        fd.write(rst.encode('utf-8') + '\n')
    fd.close()

def test():
    print 'Loading vocab...'
    cwsInfo = loadCwsInfo('./model/cws.info')
    cwsData = loadCwsData('./model/cws.data')
    print 'Done!'
    print 'Loading model...'
    #model = train(cwsInfo, cwsData, './model/cws.w2v.model', './model/cws.w2v.model.weights')
    #model = loadModel('./model/cws.w2v.model', './model/cws.w2v.model.weights')
    model = loadModel('./model/cws.model', './model/cws.model.weights')
    print 'Done!'
    print '-------------start predict----------------'
    #s = u'为寂寞的夜空画上一个月亮'
    #print cwsSent(s, model, cwsInfo)
    cwsFile('~/work/corpus/icwb2/testing/msr_test.utf8', './msr_test.utf8.cws', model, cwsInfo)

if __name__ == '__main__':
    test()