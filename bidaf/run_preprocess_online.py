# -*- coding: utf-8 -*-
import os
import argparse
from preprocess_online import process_data

parser = argparse.ArgumentParser(description='PyTorch implementation for Multiway Attention Networks for Modeling '
                                             'Sentence Pairs of the AI-Challenges')
dataRoot = './data/'
versionMark = ""
#if not os.path.exists(dataRoot+versionMark):
#    os.makedirs(dataRoot+versionMark)
parser.add_argument('--data', type=str, default=dataRoot,
                    help='location directory of the data corpus')
parser.add_argument('--threshold', type=int, default=1,
                    help='threshold count of the word')
parser.add_argument('--embed_dim', type=int, default=300,
                    help='threshold count of the word')
parser.add_argument('--char_emb_dim', type=int, default=300,
                    help='threshold count of the word')
parser.add_argument('--pretrained_embedding_path', type=str, default=dataRoot+'jwe_word2vec_size300.txt',
                    help='location directory of the data corpus')
parser.add_argument('--pretrained_char_embedding_path', type=str, default=dataRoot+'jwe_word2vec_size300.txt',
                        help='location directory of the data corpus')
parser.add_argument('--train_path', type=str, default=dataRoot+'trainingset.json',
                    help='location directory of the data corpus')
parser.add_argument('--valid_path', type=str, default=dataRoot+'validationset.json',
                    help='location directory of the data corpus')
parser.add_argument('--testa_path', type=str, default=dataRoot+'testa.json',
                    help='location directory of the data corpus')
parser.add_argument('--out_embedding_path', type=str, default=dataRoot+'embedding'+versionMark+'.table',
                    help='output embedding path')
parser.add_argument('--out_char_embedding_path', type=str, default=dataRoot+'char_embedding'+versionMark+'.table',
                    help='output embedding path')
parser.add_argument('--out_word2id_path', type=str, default=dataRoot+'word2id'+versionMark+'.table',
                    help='location directory of the data corpus')
parser.add_argument('--out_char2id_path', type=str, default=dataRoot+'char2id'+versionMark+'.table',
                    help='location directory of the data corpus')

args = parser.parse_args()

vocab_size = process_data(args.data, args.train_path,args.valid_path,args.testa_path,
                          args.threshold,args.embed_dim,args.char_emb_dim,args.pretrained_embedding_path,
                          args.out_embedding_path,args.out_char_embedding_path,args.out_word2id_path,args.out_char2id_path
                          ,versionMark
                          )

