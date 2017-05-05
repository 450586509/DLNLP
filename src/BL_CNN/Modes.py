#encoding=utf-8
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm
from keras.layers import Input,Embedding,Bidirectional,LSTM,GRU,Convolution1D,GlobalMaxPooling1D,Merge, Dense, Dropout
from HyperPara import HP
from keras.models import Model
from utils import get_logger
from keras.models import load_model
logger = get_logger(__name__)
from TextData import TextUtils,TextDataGenerator

class EarlyStop:
    '''
    以accuracy为判断标准。验证集的准确有max_cout次没有大于最大值，则停止训练。
    '''
    def __init__(self,max_count):
        self._max_accuray = 0
        self._count = -1
        self.max_cout = max_count
    def stop_train(self,acc):
        if acc < self._max_accuray:
            self._count = self._count + 1
        else:
            self._count = 0
            self._max_accuray = acc
        
        return self._count == self.max_cout


def Bi_LSTM(embedding_para={}, lstm_para={}, dense_para= {}):
    mode_in = Input(shape=(HP.MAX_LEN,), dtype='int32')
    embedding_layer = Embedding(output_dim=embedding_para['output_dim'],
                                input_dim=embedding_para['input_dim'],
                                weights=[embedding_para['weights']],
                                dropout=embedding_para['drop_out']
                                )
    embedding = embedding_layer(mode_in)
    logger.info('embedding output shape : {}'.format(embedding_layer.output_shape))
    bi_lstm = Bidirectional(LSTM(lstm_para['hidden'],
                                 return_sequences=lstm_para['return_sequences'],
                                 dropout_U=lstm_para['dropout_u'],
                                 dropout_W=lstm_para['dropout_w']))
    bi_lstm_out = bi_lstm(embedding)
    logger.info('bi_lstm output shape : {}'.format(bi_lstm.output_shape))
    mode_out = Dense(dense_para['nb_classes'], activation='softmax')(bi_lstm_out)
    model = Model(input=mode_in, output=mode_out)
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model

def CNN(embedding_para = {}, cnn_para={},dense_para = {} ):
    mode_in = Input(shape=(HP.MAX_LEN,), dtype='int32')
    embedding_layer = Embedding(output_dim=embedding_para['output_dim'],
                                input_dim=embedding_para['input_dim'],
                                weights=[embedding_para['weights']],
                                dropout=embedding_para['drop_out']
                                )
    embedding = embedding_layer(mode_in)
    conv_result = []
    for filter_length in cnn_para['filter_lengths']:
        conv = Convolution1D(nb_filter=cnn_para['filters'],
                             filter_length=filter_length,
                             border_mode=cnn_para['border_mode'],
                             activation=cnn_para['activation']
                             )(embedding)
        pooling = GlobalMaxPooling1D()(conv)
        conv_result.append(pooling)
    merge_layer = Merge(mode='concat')
    merge_out = merge_layer(conv_result)
    logger.info('(None,{0}) = {1})'.format(3 * cnn_para['filters'], merge_layer.output_shape))

    # Dropout
    merge_out = Dropout(0.5)(merge_out)

    # softmax
    mode_out = Dense(dense_para['nb_classes'], activation='softmax')(merge_out)

    # 编译模型
    model = Model(input=mode_in, output=mode_out)
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model

def BL_CNN(
        embedding_para={},
        lstm_para={},
        cnn_para={},
        dense_para={}
          ):
    mode_in = Input(shape=(HP.MAX_LEN,), dtype='int32')
    embedding_layer = Embedding(output_dim  = embedding_para['output_dim'],
                          input_dim = embedding_para['input_dim'] ,
                          weights =[embedding_para['weights']],
                          dropout=embedding_para['drop_out']
                          )
    embedding = embedding_layer(mode_in)
    logger.info('embedding output shape : {}'.format(embedding_layer.output_shape))
    # LSTM
    logger.info('dropout_u = {},dropout_w = {}'.format(lstm_para['dropout_u'], lstm_para['dropout_w']))
    bi_lstm = Bidirectional(LSTM(lstm_para['hidden'],
                                 return_sequences=lstm_para['return_sequences'],
                                 dropout_U=lstm_para['dropout_u'],
                                 dropout_W=lstm_para['dropout_w']))
    bi_lstm_out = bi_lstm(embedding)
    logger.info('bi_lstm input_shape = {}'.format( bi_lstm.input_shape))
    logger.info('bi_lstm output_shape = {}'.format(bi_lstm.output_shape))
    bi_lstm_out = Dropout(0.5)(bi_lstm_out)
    # CNN
    conv_result = []
    for filter_length in cnn_para['filter_lengths']:
        conv = Convolution1D(nb_filter = cnn_para['filters'],
                             filter_length = filter_length,
                             border_mode=cnn_para['border_mode'],
                             activation=cnn_para['activation']
                             )(bi_lstm_out)
        pooling = GlobalMaxPooling1D()(conv)
        conv_result.append(pooling)
    merge_layer = Merge(mode='concat')
    merge_out = merge_layer(conv_result)
    logger.info('(None,{0}) = {1})'.format(3 * cnn_para['filters'], merge_layer.output_shape))

    # Dropout
    merge_out = Dropout(0.5)(merge_out)

    # softmax
    mode_out = Dense(dense_para['nb_classes'], activation='softmax')(merge_out)

    # 编译模型
    model = Model(input=mode_in, output=mode_out)
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=HP.PATIENCE, verbose=1),
        # ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]
    return model, callbacks 
if __name__ =="__main__":
    tdg = TextDataGenerator(max_len=HP.MAX_LEN,
                            language=HP.LANGUAGE,
                            is_char=HP.IS_CHAR,
                            classes = HP.CLASSES
                            )
    tdg.build_word2index(filename_list=HP.WORD2INDEX_FILES)
    
    embedding_weights = TextUtils.embedding_weights(word2index = tdg.word_index,
                                embedding_source=HP.GOOGLE_PRETRAINED_VECTOR,
                                dimmension=HP.DIMMENSION,
                                trainded_model = 'google_w2v')
    logger.info('embedding layer weights shape ={}'.format(embedding_weights.shape))
    
    embdedding={'weights':embedding_weights,'drop_out':0,'input_dim':len(embedding_weights),'output_dim':300}
    lstm_para  ={'hidden':300,'return_sequences':True,'dropout_u':0.5,'dropout_w':0.5}
    cnn_para = {'filter_lengths':[3,4,5],'filters':100,'activation':'relu','border_mode':'same'}
    dense_para={'nb_classes':5}
    
    dev_X,dev_y = tdg.load_array_data(files_names = HP.DEV_DATA_FILES)
    test_X,test_y = tdg.load_array_data(files_names=HP.TEST_DATA_FILES)
    logger.info('dev_X shape = {},dev_y_shape ={}'.format(dev_X.shape,dev_y.shape))
    logger.info('test_X shape = {},test_y_shape ={}'.format(test_X.shape,test_y.shape))
    all_test_result =[]
    for run_time in range(HP.RUN_TIMES):
        logger.info('epoch per run:{} run_time :{} / total run_time:{}'.format(HP.EPOCHES,run_time,HP.RUN_TIMES))
        bl_cnn, callbacks = BL_CNN(embedding_para=embdedding,
                                   lstm_para=lstm_para,
                                   cnn_para=cnn_para,
                                   dense_para=dense_para)
        best_test_accu = 0

        for batch in range(HP.EPOCHES):
            for batch_x,batch_y in tdg.files_generator(file_names=HP.TRAIN_DATA_FILES,
                                               batch_size=50):
                bl_cnn.train_on_batch(batch_x,batch_y)
            else:
                loss,accu = bl_cnn.evaluate(x=dev_X,y=dev_y)
                logger.info('eval loss = {}, accuracy ={}'.format(loss,accu))
                if accu > best_test_accu:
                    bl_cnn.save('./best_model.h5')
                    best_test_accu = accu 
                    logger.info('saved model')
        else:
            del bl_cnn
            bl_cnn = load_model('./best_model.h5')
            test_loss,test_accu = bl_cnn.evaluate(x = test_X, y = test_y)
            logger.info('test loss ={}, accuracy ={}'.format(test_loss,test_accu))
            all_test_result.append(test_accu)
            del bl_cnn
    logger.info('sst-5 bl_cnn test result:{}'.format(all_test_result))

