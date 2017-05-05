#encoding=utf-8
class HP:
    def __init__(self):
        pass
    # 数据预处理参数
    MAX_LEN = 120
    LANGUAGE = 'english'
    IS_CHAR = False
    PATH = '/home/bruce/data/sentiment/origin/'
    WORD2INDEX_FILES=[PATH+'stsa.fine.train',
                      PATH+'stsa.fine.dev',
                      PATH+'stsa.fine.test']
    TRAIN_DATA_FILES=[PATH+'stsa.fine.train']
    DEV_DATA_FILES = [PATH+'stsa.fine.dev']
    TEST_DATA_FILES = [PATH + 'stsa.fine.test']
    GOOGLE_PRETRAINED_VECTOR = '/home/bruce/data/google_word2vec/GoogleNews-vectors-negative300.bin'
    DIMMENSION=300
    CLASSES = 5

    # TRAIN
    PATIENCE = 10
    RUN_TIMES = 10
    EPOCHES = 24


