#encoding=utf-8
import Modes
from utils import get_logger
from sklearn.cross_validation import  StratifiedKFold
import numpy as np
from keras.models import load_model
from TextData import TextUtils,TextDataGenerator
import ResultAnalysis
logger = get_logger(__name__)

if __name__=='__main__':
    nb_class = 2
    model_file = './chinese_couprse_best_model.h5'
    total_test_acc =[]
    tdg = TextDataGenerator(max_len = 300,
                            language='chinese',
                            is_char=True,
                            classes = nb_class
                            )
    embedding_file =  '/home/bruce/data/polyglot-zh_char.pkl'
    data_file = ['/home/bruce/data/6000.txt']
    tdg.build_word2index(filename_list = data_file)
    
    embedding_weights = TextUtils.embedding_weights(word2index = tdg.word_index,
                                embedding_file=embedding_file,
                                dimension=64,
                                trainded_model = 'polyglot')

    logger.info('embedding layer weights shape ={}'.format(embedding_weights.shape))
    
    embdedding={'weights':embedding_weights,'drop_out':0,'input_dim':len(embedding_weights),'output_dim':64}
    lstm_para  ={'hidden':64,'return_sequences':True,'dropout_u':0.5,'dropout_w':0.5}
    cnn_para = {'filter_lengths':[3,4,5],'filters':100,'activation':'relu','border_mode':'same'}
    dense_para={'nb_classes':nb_class}
    X,y = tdg.load_array_data(files_names = data_file)
    index = np.array(list(range(len(y))))
    np.random.shuffle(index)
    X = X[index]
    y = y[index]
    y_label = [np.argmax(v) for v in y]
    logger.info('y[0:10] = {}'.format(y[0:10]))
    logger.info('y_label[0:10] = {}'.format(y_label[0:10]))
    logger.info('dev_X shape = {},dev_y_shape ={}'.format(X.shape,y.shape))
    all_test_result =[]
    kfold = StratifiedKFold(y=y_label,n_folds=10)
    for k,(train_index,test_index) in enumerate(kfold):
        X_train  = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        middle = int(0.9*len(X_train))
        X_train,X_dev = X_train[:middle],X_train[middle:]
        y_train,y_dev = y_train[:middle],y_train[middle:]
        max_val_acc = 0
        logger.info('X_train shape:{}, y_train shape:{};X_dev shape:{} ,y_dev shape:{};X_test shape:{},y_test shape:{}'.format(X_train.shape,y_train.shape,X_dev.shape,y_dev.shape,X_test.shape,y_test.shape))
        bl_cnn, callbacks = Modes.BL_CNN(embedding_para=embdedding,
                                   lstm_para=lstm_para,
                                   cnn_para=cnn_para,
                                   dense_para=dense_para)
        max_val_acc = 0 
        #early stopping
        earlystop = Modes.EarlyStop(max_count = 20)
        for epoch in range(200):
            history_data = bl_cnn.fit(X_train,
                                     y_train,
                                     batch_size=50,
                                     nb_epoch=1,
                                     validation_data=(X_dev, y_dev)
                                     )
            val_acc = history_data.history['val_acc'][0]
            train_acc = history_data.history['acc'][0]
            train_loss = history_data.history['loss'][0]
            val_loss = history_data.history['val_loss'][0]
            logger.info('runtime = {} epoch ={},train_acc={},train_loss={},val_acc={},val_loss={}'.format(k,epoch+1, train_acc, train_loss,val_acc, val_loss))
            if val_acc > max_val_acc:
                logger.info('saving model')
                max_val_acc = val_acc
                bl_cnn.save(model_file)
            if earlystop.stop_train(val_acc):
                logger.info('early stopping at epoch = {}'.format(epoch))
                break
            logger.info('_count = {},_max_accuray]={} '.format(earlystop._count,earlystop._max_accuray))
        #testing
        logger.info('testing model')
        del bl_cnn  # deletes the existing model
        bl_cnn = load_model(model_file)
        score, acc = bl_cnn.evaluate(X_test, y_test, batch_size=50, verbose=1, sample_weight=None)
        logger.info('max_val_acc :{}'.format(max_val_acc))
        logger.info('Test score:{}'.format( score))
        logger.info('Test accuracy: {}'.format(acc))
        total_test_acc.append(acc)
        #get predict_y
        predictions = bl_cnn.predict(X_test)
        logger.info('predictions[0:10] = {}'.format(predictions[0:10]))
        predict_y = np.ndarray.flatten(np.argmax(predictions, axis= 1 ))
        logger.info('predict_y[0:10]={}'.format(predict_y[0:10]))
        #export result
        index_word = dict([(kv[1],kv[0])for kv in tdg.word_index.items()])
        ResultAnalysis.comp_predict_lable(X = X_test,y=y_test,y_predict = predict_y,
                                          index2word=index_word,label2name={0:'negtive',1:'positive'},
                                          file_to_save='./chinese_zi_epoch_{}.txt'.format(k))

    logger.info('total_test_acc={}'.format(total_test_acc))
    logger.info('mean acc = {}'.format(sum(total_test_acc) / len(total_test_acc)))
    logger.info('max acc = {}'.format( max(total_test_acc)))     

  



