#coding=utf-8
#读去数据，并且分词。
from os.path import join
import jieba
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
import numpy as np
def get_train_data(path=''):
    X=[]
    y=[]
    with open(join(path,'pos.txt'),'r') as fr:
        for line in fr.readlines():
            X.append(' '.join(jieba.cut(line,cut_all=False)))
            y.append(1)
    with open(join(path,'neg.txt'),'r') as fr:
        for line in fr.readlines():
            X.append(' '.join(jieba.cut(line,cut_all=False)))
            y.append(0)
    return X,y

def bagOfWord(X):
    vectorizer = CountVectorizer(min_df=8, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(X)
    with open('./model/vectorizer.pkl','wb') as fr:
        print('save text vectorizer to ./model/')
        pickle.dump(vectorizer,fr)
    return X

def train_model(X=[],y=[],model=''):
    if model == 'GaussianNB':
        bayes = GaussianNB()
    elif model == 'Bernoulli':
        bayes = BernoulliNB()
    else:
        bayes = MultinomialNB()
    bayes.fit(X, y)
    print('saving bayes model to ./model')
    with open('./model/bayes.pkl','wb') as fr:
        pickle.dump(bayes,fr)

def train(train_path='',):
    X,y = get_train_data(path=train_path)
    X = bagOfWord(X)
    train_model(X,y)
def predict_sentence(s=''):
    with open('./model/vectorizer.pkl','rb') as f:
        vectorizer = pickle.load(f)
    with open('./model/bayes.pkl','rb') as f:
        bayes = pickle.load(f)
    s =[' '.join(jieba.cut(s, cut_all=False))]
    x = vectorizer.transform(s)
    predict = bayes.predict(x)
    print(predict)
    if predict ==1:
        print('positive')
    else:
        print('negtive')
#train(train_path = 'G:\\code\\DLNLP\\src\\data')
predict_sentence('这个东西感觉太贵了，我很满意')