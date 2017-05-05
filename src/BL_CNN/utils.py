import codecs
import numpy as np
import logging

def get_logger(file_name):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG)
    ## create console handler(handler resposible for deal loggin and send to different place)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s" )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    ## add handler to logger
    fh = logging.FileHandler('/home/bruce/data/log/cnn.log',encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger

