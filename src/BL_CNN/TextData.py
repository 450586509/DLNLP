import random
import gensim
from utils import get_logger
from collections import defaultdict
import re
import numpy as np
from HyperPara import HP
import pickle

logger = get_logger(__name__)


class TextUtils:
    def __init__(self):
        pass
    
    def clean_str_chinese(string):
        string = string.strip()
        return string

    def seg_sentence(sentence,cut_all = False):
        import jieba
        result = jieba.cut(sentence,cut_all=cut_all)
        return list(result)

    def clean_str_english(string):

        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()
    def embedding_weights( word2index, embedding_file,dimension,trainded_model):
        '''
        :param word2index: word2index dictionary
        :param embedding_file:  a file contain word and its embedding
        :param trainded_model: 'google_w2v','polyglot'
        :return:
        '''
        logger.info('loading pre-trained word vector from : {}'.format(embedding_file))
        if trainded_model == 'google_w2v':
            word_embedding_dict = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
        elif trainded_model == 'polyglot':
            words, embeddings = pickle.load(open(embedding_file, 'rb'), encoding='latin1')
            word_embedding_dict = dict(zip(words,embeddings))
        else:
            logger.error('the {} train embedding model is not exist ')
            raise
        weights = np.zeros((len(word2index) + 1, dimension))

        logger.info('loading finished')
        count_in = 0
        logger.info('building Embedding weights ')
        for word , index in word2index.items():
            if word in word_embedding_dict:
                count_in = count_in + 1
                weights[index] = word_embedding_dict[word]
            else:
                if trainded_model == 'google_w2v':
                    weights[index] = np.random.uniform(-0.25,0.25,size=dimension)
                elif trainded_model == 'polyglot':
                    weights[index] = word_embedding_dict['<UNK>']
        weights[0] = np.random.uniform(-0.25,0.25,size=dimension)
        logger.info('finish building')
        logger.info('embedding weights shape :{}'.format(weights.shape))
        total_words = len(word2index)
        logger.info('words pre-trained/not-pretrained/total-words : {}/{}/{}  '.format(count_in,total_words-count_in,total_words))
        return weights 
class TextDataGenerator:
    def __init__(self,
                 max_len,
                 language,
                 is_char,
                 classes,
                 batch_size=32,
                 encoding='utf-8',
                 ):
        self.encoding = encoding
        self.batch_size = batch_size
        self.is_char = is_char
        self.language = language
        self.max_len = max_len
        self.word_count = defaultdict(int)
        self.word_index = {}
        self.classes = classes
        assert self.batch_size >=1,"batch_size is 0"
        assert self.max_len>=1
    def build_wordcount(self,
                        filenames = []
                        ):
        for filename in filenames:
            logger.info("readling filename = {}".format(filename))
            for line in open(filename, encoding = self.encoding):
                line = line[2:]
                line = TextUtils.clean_str_english(line) if self.language == 'english' else TextUtils.clean_str_chinese(line)
                if self.language == 'english':
                    words =  list(line) if self.is_char else line.split()
                elif self.language == 'chinese':
                    words = list(line) if self.is_char else TextUtils.seg_sentence(line)
                for i, word in enumerate(words, start=1):
                    if i > self.max_len:
                        break
                    self.word_count[word] = self.word_count[word] + 1

    def build_word2index(self,
                         filename_list,
                         ):
        import operator
        self.build_wordcount(filenames=filename_list
                            )
        sorted_wc = sorted(self.word_count.items(), key=operator.itemgetter(1), reverse=True)
        for index, wc in enumerate(sorted_wc, start=1):
            self.word_index[wc[0]] = index
        logger.info('size of wordindex :{}'.format(len(self.word_index)))

    def to_index(self,words):
        return [self.word_index[word] if word in self.word_index else 0 for word in words ]
    def padding(self,sentnece):
        if self.language == 'english':
            words = list(sentnece) if self.is_char else sentnece.split()
        elif self.language == 'chinese':
            words = list(sentnece) if self.is_char else TextUtils.seg_sentence(sentnece)
        else:
            logger.error('wrong language')
            raise Exception
        if len(words) >= self.max_len:
            return words[0:self.max_len]
        else:
            zeros = [0 for i in range(0,self.max_len-len(words))]
            return words + zeros
    def to_categorical(self,batch_y):
        zeros = np.zeros((len(batch_y),self.classes))
        for index,label in enumerate(batch_y):
            zeros[index][label] = 1
        return zeros


    def files_generator(self, file_names,batch_size):
        self.batch_size = batch_size
        batch_x = []
        batch_y = []
        count = 0
        total_lines = []
        for input_file in file_names:
            with open(input_file, encoding=self.encoding) as fr:
                lines = fr.readlines()
                total_lines = total_lines + lines
        random.shuffle(total_lines)
        logger.info("shuffer data , {} line ".format(len(total_lines)))
        for line in total_lines:
            line = TextUtils.clean_str_english(line) if self.language == "english" \
                else TextUtils.clean_str_chinese(line)
            if len(line) <= 1 :
                continue
            label = int(line[0:1])
            sentence = line[2:]
            words = self.padding(sentence)
            sentence_index = self.to_index(words)
            batch_x.append(sentence_index)
            batch_y.append(label)
            count = count + 1
            if self.batch_size == count:
                yield np.array(batch_x), self.to_categorical(batch_y)
                batch_x = []
                batch_y = []
                count = 0
        else:
            if len(batch_x) >= 1:
                yield np.array(batch_x), self.to_categorical(batch_y)
    def load_array_data(self,files_names):
        X = []
        y = []
        total_lines = []
        for input_file in files_names:
            with open(input_file, encoding=self.encoding) as fr:
                lines = fr.readlines()
                total_lines = total_lines + lines
        logger.info("loaded data , {} line ".format(len(total_lines)))
        for line in total_lines:
            line = TextUtils.clean_str_english(line) if self.language == "english" \
                else TextUtils.clean_str_chinese(line)
            if len(line) <= 1 :
                continue
            label = int(line[0:1])
            sentence = line[2:]
            words = self.padding(sentence)
            sentence_index = self.to_index(words)
            X.append(sentence_index)
            y.append(label)
        return np.array(X),self.to_categorical(y)




if __name__ == "__main__":
    logger.info('-' * 20 + 'begin' + '-' * 20)
    tdg = TextDataGenerator(batch_size=1,
                            language=HP.LANGUAGE,
                            encoding='utf-8',
                            is_char=HP.IS_CHAR,
                            max_len=HP.MAX_LEN)
    tdg.build_word2index(filename_list=HP.WORD2INDEX_FILES)
    '''
    TextUtils.embedding_weights(word2index=tdg.word_index,
                                embedding_source=HP.GOOGLE_PRETRAINED_VECTOR,
                                dimmension=HP.DIMMENSION)

    file_generator = tdg.files_generator(file_names=HP.DATA_FILES)
    for x,y in file_generator:
        logger.info('x = {}, y ={}'.format(x,y))
    '''
    X,y = tdg.load_array_data(files_names=HP.DEV_DATA_FILES,classes=HP.CLASSES)
    logger.info('shape X :{0} shape Y:{1}'.format(X.shape,y.shape))
    logger.info('-' * 20 + 'end' + '-' * 20)


