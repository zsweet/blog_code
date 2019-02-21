# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module prepares and runs the whole system.
"""
import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging

from model    import RCModel


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--classPointMark', action='store_true',default=True,
                        help='the mode classification(true) or pointer-network(false)')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--getSoftmax', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='2',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=3e-7,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.7,
                                help='dropout keep rate')
    train_settings.add_argument('--dropout_embedding_prob', type=float, default=0.9,
                                           help='dropout embedding rate')
    train_settings.add_argument('--batch_size', type=int, default=128,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=20,
                                help='train epochs')



    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=100,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=30,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')
    model_settings.add_argument('--max_char_length_in_word',type=int,default=4)

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/demo/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/demo/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/demo/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')


    ######zsw  M : 09-29
    log_dir = "./summary"
    parser.add_argument('--data', type=str, default='data/',help='location directory of the data corpus')
    parser.add_argument('--pretrained_embedding_path',type=str,default = "../../data/embedding.table")
    parser.add_argument('--pretrained_char_embedding_path',type=str,default = "../../data/char_embedding.table")
    parser.add_argument('--train_data_path', type=str, default="../../data/train.pickle")
    parser.add_argument('--dev_data_path', type=str, default="../../data/dev.pickle")
    parser.add_argument('--testa_data_path', type=str, default="../../data/testa_forTest.pickle")
    parser.add_argument('--model_prefix',type=str,default="")
    parser.add_argument('--mode', type=str,default="train",help="train | test_ | forward_mode [test]")
    parser.add_argument("--is_train", type=bool,default=True, help="train [true]")
    parser.add_argument("--use_char_emb", type=bool,default=True, help="use char emb? [True]")
    parser.add_argument("--use_word_emb", type=bool,default=True, help="use word embedding? [True]")
    parser.add_argument("--use_highway",type=bool,default= True, help="Use highway? [True]")
    parser.add_argument("--highway_num_layers", type = int ,default=2,help= "highway num layers [2]")
    parser.add_argument("--out_channel_dims", type=str,default="100",help= "Out channel dims of Char-CNN, separated by commas [100]")
    parser.add_argument("--filter_heights", type=str,default="4", help="Filter heights of Char-CNN, separated by commas [5]")
    parser.add_argument("--share_cnn_weights",type=bool,default= True, help="Share Char-CNN weights [True]")
    parser.add_argument("--share_lstm_weights", type=bool , default=True, help="Share pre-processing (phrase-level) LSTM weights [True]")
    parser.add_argument("--logit_func", type = str,default="tri_linear", help="logit func [tri_linear]")
    parser.add_argument("--end_learning_rate",type = float,default= 0.0001, help="Learning rate")
    #parser.add_argument("--grad_clip",type = float, default=5.0, help="Global Norm gradient clipping rate")
    parser.add_argument("--log_dir", type = str, default= log_dir, help = "Directory for tf event")
    parser.add_argument("--softmax_log_output_path", type=str, default="./softmax_output/dropout_embedding_rnn_cosin_valid.softmax", help="path for softmax output")
    parser.add_argument('--softmax_mode', type=str, default="valid")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return parser.parse_args()



def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")

    #加载字典数据
    logger.info('Load vocab and embedding from text...')
    with open(args.pretrained_embedding_path, 'rb') as f:
        embedding = pickle.load(f)

    logger.info('Load vocab and embedding from text...')
    with open(args.pretrained_char_embedding_path, 'rb') as f:
         char_embedding = pickle.load(f)


    #loading data
    logger.info('loading the data...')
    with open(args.train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    logger.info('train data size:' + str(len(train_data)))
    with open(args.dev_data_path, 'rb') as f:
        dev_data = pickle.load(f)
    logger.info('dev data size:' + str(len(dev_data)))


    #shuffle数据
    logger.info('shuffle the data...')
    #trainShuffleData = shuffle_data(train_data,'pWordId' )
    trainShuffleData = train_data

    logger.info('Initialize the model...')
    rc_model = RCModel(args,embedding,char_embedding)
    logger.info('Training the model...')
    rc_model.train(trainShuffleData, dev_data, args.epochs, args.batch_size,
                   save_dir=args.model_dir, save_prefix=args.algo+args.model_prefix,dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def getSoftmax(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")

    #加载字典数据
    logger.info('Load vocab and embedding from text...')
    with open(args.pretrained_embedding_path, 'rb') as f:
        embedding = pickle.load(f)

    logger.info('Load vocab and embedding from text...')
    with open(args.pretrained_char_embedding_path, 'rb') as f:
         char_embedding = pickle.load(f)


    #loading data
    logger.info('loading the data...')
    with open(args.train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    logger.info('train data size:' + str(len(train_data)))
    with open(args.dev_data_path, 'rb') as f:
        dev_data = pickle.load(f)

    with open(args.testa_data_path, 'rb') as f:
        test_data = pickle.load(f)
    logger.info('dev data size:' + str(len(dev_data)))
    logger.info('testa data size:'+str(len(test_data)))

    #shuffle数据
    logger.info('shuffle the data...')
    #trainShuffleData = shuffle_data(train_data,'pWordId' )
    #trainShuffleData = train_data

    logger.info('Initialize the model...')
    rc_model = RCModel(args,embedding,char_embedding)
    logger.info("loading model from {}{}".format(args.model_dir,args.algo+args.model_prefix))
    rc_model.restore(args.model_dir,args.algo+args.model_prefix)
    logger.info('Training the model...')
    if args.softmax_mode == 'valid':
        rc_model.get_softmax_result(train_data=dev_data, batch_size=args.batch_size,
                                dropout_keep_prob=args.dropout_keep_prob, outputPath=args.softmax_log_output_path)
    else:
        rc_model.get_softmax_result(train_data=test_data, batch_size=args.batch_size,
                                    dropout_keep_prob=args.dropout_keep_prob, outputPath=args.softmax_log_output_path)
    logger.info('Done with model training!')



def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.train:
        train(args)
    if args.getSoftmax:
        getSoftmax(args)

if __name__ == '__main__':
    run()
