#encoding=utf-8
import numpy as np
from itertools import chain
from utils import get_logger
logger = get_logger(__name__)
def comp_predict_lable(X,y,y_predict,index2word,label2name,file_to_save = None,encoding='utf-8'):
    '''
    :param X: numpy object
    :param y: y is categorical,one-hot
    :param y_predict: as y
    :return:
    '''
    assert len(y) == len(y_predict)
    y = [np.argmax(i) for i in y]
    y_predict = [np.argmax(i) for i in y_predict]
    right_predict_set = []
    wrong_predict_set = []
    X_y = zip(X,y)
    for k,v in zip(X_y,y_predict):
        if k[1] == v:
            right_predict_set.append((k,v))
        else:
            wrong_predict_set.append((k,v))
    # group with y
    right_predict_set = sorted(right_predict_set,key = lambda t : t[0][1])
    wrong_predict_set = sorted(wrong_predict_set, key=lambda t: t[0][1])
    logger.info('right_predict_set size = {}'.format(len(right_predict_set)))
    logger.info('wrong_predict_set size = {}'.format(len(wrong_predict_set)))

    if file_to_save is not None:
        with open(file_to_save,'w',encoding='utf-8')as fw:
            for data in chain(right_predict_set,wrong_predict_set):
                line = [index2word[i] for i in data[0][0] if i in index2word]
                line.append(label2name[data[0][1]])
                line.append(label2name[data[1]])
                line = ' '.join(line) +'\n'
                fw.write(line)
    data = chain(right_predict_set,wrong_predict_set)
    return [list(t[0][0])+[t[0][1]]+[t[1]] for t in data]

if __name__ == "__main__":
    X= np.array([ [1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    y =np.array([[0,1],[0,1],[1,0],[1,0]])
    y_predict = np.array([[0,1],[1,0],[1,0],[0,1]])
    index2word={1:'我',2:'喜',3:'欢',4:'乐'}
    label2name={0:'neg',1:'pos'}
    comp_predict_lable(X,y,y_predict,index2word=index2word,label2name= label2name,file_to_save='./analysis.txt')




