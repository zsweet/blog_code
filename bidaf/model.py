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
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import random
import logging
import tensorflow as tf
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from utils.zsw_model_util import multi_conv1d,cosine_decay_restarts
from utils.zsw_model_util import highway_network,bidirectional_dynamic_rnn,attention_layer
from tensorflow.python.ops.rnn_cell import BasicLSTMCell   ###!!!!!!!!!!!!!!!!注意修改
import math
from utils.zsw_util import *
import json

class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self,args,embedding,char_emb):
        self.classPointMark = args.classPointMark
       # print (">>>>>>>>>>>>>>>>{}".format(self.classPointMark))
        # logging
        self.logger = logging.getLogger("brc")

        #train or else
        #self.is_train = args.is_train

        # basic config
        self.batch_size = args.batch_size
        self.algo = args.algo
        self.mode = args.mode
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1
        self.dropout_embedding = args.dropout_embedding_prob
        self.use_char_emb = args.use_char_emb
        self.use_word_emb = args.use_word_emb
        self.use_highway = args.use_highway
        self.highway_num_layers = args.highway_num_layers
        self.wd = self.weight_decay
        self.out_channel_dims = args.out_channel_dims  #char embedding cnn filter size
        self.filter_heights = args.filter_heights
        self.share_cnn_weights = args.share_cnn_weights
        self.share_lstm_weights = args.share_cnn_weights
        self.logit_func = args.logit_func
        self.log_dir = args.log_dir+'/BIDAF'+args.model_prefix
        self.end_learning_rate = args.end_learning_rate
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.num_steps = math.ceil(249433 / self.batch_size)
        # attention method type
        #self.dynamic_att = args.dynamic_att

        # length limit
#        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
 #       self.max_a_len = args.max_a_len
        self.word_width = args.max_char_length_in_word

        #embedding tables
        self.word_embeddings = embedding
        self.char_emb = char_emb

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())
        self.var_list = tf.global_variables()
        self.print_variable()

    def print_variable(self):
        print("---------- Model Variabels -----------")
        cnt = 0
        for var in self.var_list:
            cnt += 1
            try:
                var_shape = var.get_shape()
            except:
                var_shape = var.dense_shape
            str_line = str(cnt) + '. ' + str(var.name) + '\t' + str(var.device) + '\t' + str(var_shape) + '\t' + str(
                var.dtype.base_dtype)
            print(str_line)
        print('------------------------')

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._zsw_attention()
        if self.classPointMark:
            self._yesno()


        # self.sess.run(tf.global_variables_initializer())
        # self.var_list = tf.global_variables()
        # self.print_variable()

        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))


    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [self.batch_size,self.max_p_len])
        self.p_mask = tf.placeholder(tf.bool,[self.batch_size,self.max_p_len])
        self.cp = tf.placeholder(tf.int32,[self.batch_size,self.max_p_len,self.word_width])
        self.q = tf.placeholder(tf.int32, [self.batch_size, self.max_q_len])
        self.q_mask = tf.placeholder(tf.bool,[self.batch_size,self.max_q_len])
        self.cq = tf.placeholder(tf.int32,[self.batch_size,self.max_q_len,self.word_width])
        self.p_length = tf.placeholder(tf.int32, [self.batch_size])
        self.q_length = tf.placeholder(tf.int32, [self.batch_size])
        self.cp_length = tf.placeholder(tf.int32, [self.batch_size])
        self.cq_length = tf.placeholder(tf.int32, [self.batch_size])
        self.answer = tf.placeholder(tf.float32,[None,3])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        #self.is_train = tf.placeholder(tf.bool)#, [], name='is_train')

    def _zsw_attention(self):
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        with tf.variable_scope("emb"):
            if self.use_char_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    print ("char_emb :{}*{}".format(len(self.char_emb),len(self.char_emb[0])))
                    char_emb_mat = tf.get_variable("char_embedding",
                                                   shape=[len(self.char_emb),len(self.char_emb[0])],
                                                   initializer=tf.constant_initializer(self.char_emb),
                                                   dtype=tf.float32,
                                                   trainable=False)
                ### batch
                with tf.variable_scope("char"):
                    Acx = tf.nn.embedding_lookup(char_emb_mat, self.cp)  # [batch, doc_size,char_limit_size,char_embedding_size]
                    Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [batch, query_size,char_limit_size, char_embedding_size]
                    #Acx = tf.reshape(Acx, [-1, JX, W, dc])
                    #Acq = tf.reshape(Acq, [-1, JQ, W, dc])
                    Acx = tf.nn.dropout(Acx, self.dropout_embedding)
                    Acq = tf.nn.dropout(Acq, self.dropout_embedding)

                    filter_sizes = list(map(int, self.out_channel_dims.split(',')))
                    heights = list(map(int, self.filter_heights.split(',')))
                    #assert sum(filter_sizes) == dco, (filter_sizes, dco)

                    with tf.variable_scope("conv"):
                        xx = multi_conv1d(Acx, filter_sizes, heights, "VALID", keep_prob=self.dropout_keep_prob,scope="xx")
                        if self.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", keep_prob= self.dropout_keep_prob,scope="xx")
                        else:
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", keep_prob=self.dropout_keep_prob,scope="qq")
                        #xx = tf.reshape(xx, [-1, M, JX, dco])
                        #qq = tf.reshape(qq, [-1, JQ, dco])

            if self.use_word_emb :
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    #if self.mode == 'train_mode':
                        word_emb_mat = tf.get_variable("word_emb_mat",
                                                       shape=[len(self.word_embeddings),len(self.char_emb[0])],
                                                       initializer=tf.constant_initializer(self.word_embeddings),
                                                       dtype = tf.float32,
                                                       trainable=False)
                    #else:
                     #   word_emb_mat = tf.get_variable("word_emb_mat",
                      #                                 shape=[len(self.word_embeddings), len(self.char_emb[0])],
                      #                                 dtype=tf.float32)

                with tf.name_scope("word"):
                    Ax = tf.nn.embedding_lookup(word_emb_mat, self.p)  # [batch_size,doc_char_size,word_emb_size]
                    Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  #[batch_size,query_char_size,word_emb_size]i
                    Ax = tf.nn.dropout(Ax,self.dropout_embedding)
                    Aq = tf.nn.dropout(Aq,self.dropout_embedding)
                if self.use_char_emb:
                    xx = tf.concat([xx, Ax], 2)  # [batch_size,doc_char_size,word_emb_size]
                    qq = tf.concat([qq, Aq], 2)  #[batch_size,query_char_size,word_emb_size]
                else:
                    xx = Ax
                    qq = Aq
        print("xx_first:", xx)
        # highway network
        if self.use_highway:
            with tf.variable_scope("highway"):
                xx = highway_network(xx, self.highway_num_layers, True, wd=self.wd,input_keep_prob=self.dropout_keep_prob)
                tf.get_variable_scope().reuse_variables()
                qq = highway_network(qq, self.highway_num_layers, True, wd=self.wd,input_keep_prob=self.dropout_keep_prob)
        print("xx_second:", xx)

        #cell = BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        #d_cell = BasicLSTMCell(self.hidden_size,state_is_tuple=True)  # SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
        x_len = self.p_length  #tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = self.q_length  #tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]
        #print("encode q:==========u1 para==========: d_cell=={},qq=={},q_len=={}".format(d_cell, qq, q_len))
        #print("encode p:==========u1 para==========: d_cell=={},qq=={},q_len=={}".format(cell, xx, x_len))
        with tf.variable_scope("prepro"):
            #(fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = bidirectional_dynamic_rnn(d_cell, d_cell, qq, q_len,
             #                                                                    dtype=tf.float32,
              #                                                                   scope='u1')  # [N, J, d], [N, d]
           # u = tf.concat([fw_u, bw_u], 2)
            with tf.variable_scope('query_encoding'):
              u,_ = rnn('bi-lstm',  qq , q_len, self.hidden_size,dropout_keep_prob=self.dropout_keep_prob)
              #u = tf.nn.dropout(u,self.dropout_keep_prob)
            if self.share_lstm_weights:    # 这个地方修改成没有用了~
                #tf.get_variable_scope().reuse_variables()
                #(fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype=tf.float32,
                                          #                  scope='u1')  # [N, M, JX, 2d]
                #h = tf.concat([fw_h, bw_h], 2)  # [N, M, JX, 2d]
                with tf.variable_scope('passage_encoding'):
                    h,_ = rnn('bi-lstm',  xx , x_len, self.hidden_size,dropout_keep_prob=self.dropout_keep_prob)
                   # h = tf.nn.dropout(h,self.dropout_keep_prob)
            else:
#                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype=tf.float32,
 ##                                                           scope='h1')  # [N, M, JX, 2d]
   #             h = tf.concat([fw_h, bw_h], 2)  # [N, M, JX, 2d]
                 with tf.variable_scope('passage_encoding'):
                       h,_ = rnn('bi-lstm',  xx , x_len, self.hidden_size,dropout_keep_prob=self.dropout_keep_prob)

        with tf.variable_scope("main"):
            # if self.dynamic_att:
            #     p0 = h
            #     u = tf.reshape(tf.tile(tf.expand_dims(u, 1), [1, M, 1, 1]), [N * M, JQ, 2 * d])
            #     q_mask = tf.reshape(tf.tile(tf.expand_dims(self.q_mask, 1), [1, M, 1]), [N * M, JQ])
            #     first_cell = AttentionCell(cell, u, mask=q_mask, mapper='sim',
            #                                input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
            # else:
            #h_expand = tf.expand_dims(h,1)
            #p_mask_expand = tf.expand_dims(self.p_mask,1)
            #print("h_expand:{} p_mask_expand:{}".format(h_expand,p_mask_expand))
            #p0 = attention_layer(self.logit_func, self.wd, h_expand,u,h_mask=p_mask_expand , u_mask=self.q_mask, scope="p0")
            #p0 = tf.squeeze(p0,1)
           # first_cell = d_cell
            # p0 = h
            # first_cell = cell #!!!!!!!!!!!!!!!!!!!!!!!!!!
            N = tf.shape(0)
            J = tf.shape(u)[1]
            h_expand = tf.tile(tf.expand_dims(h,2),[1,1,J,1]) ##[N,T,J,d]  将context每个词复制J份
            T = tf.shape(h)[1]
            u_expand = tf.tile(tf.expand_dims(u,1),[1,T,1,1])  ##[N,T,J,d]  将query整体复制T份
            h_element_wise_u=tf.multiply(h_expand, u_expand)  ##[N,T,J,d]  [N,context中每个词与query整体的mutiply结果,context中某个词与query每个词的mutiply结果,context中某个词与query某个词的mutiply结果]
            cat_data = tf.concat((h_expand,u_expand,h_element_wise_u),3)  #[N,T,J,3d]
#            S = tf.layers.dense(tf.reshape(cat_data,[tf.shape(self.p)[0],-1]),1).reshape(N,T,J)
            print("cat_data:{}".format(cat_data))
            S = tf.layers.dense(cat_data,1)  ##[N,T,J,1]
            S = tf.nn.dropout(S,self.dropout_keep_prob)   #！！！！！！！
            print("S:{}".format(S))
            S = tf.squeeze(S,3)   ##[N,T,J]
            print("S reshape result:{}".format(S))
            # Context2Query
            c2q = tf.matmul(tf.nn.softmax(S),u)  # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
            # Query2Context
            # b: attention weights on the context
            b = tf.nn.softmax(tf.reduce_max(S, 2), dim=-1)  #N*T  和query中每个词关系度比较大的context词的分数(这里面包含了上下文语义)，
                                                            # 这里可能不止一个(具体是找出context中每个词最相关的query)，
                                                            # 然后softmax，并求和
            q2c = tf.matmul(tf.expand_dims(b,1), h) # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
            print("q2c shape : {}".format(q2c)) 
            q2c = tf.tile(q2c,[1,T,1])    # (N, T, 2d)
            #q2c.set_shape([self.batch_size,self.max_p_len,2*self.hidden_size])
            print("q2c expand_dim shape : {}".format(q2c)) 
            G = tf.concat((h, c2q, tf.multiply(h,c2q), tf.multiply(h,q2c)), 2)  # (N, T, 8d)
            print("G shape:{},\nh shape:{}\nc2q shape:{}\nh multiply c2q:{}\nh multiply q2c:{}".format(G,h, c2q, tf.multiply(h,c2q), tf.multiply(h,q2c)))

    #        print("decode==========g0 para==========: first_cell:{},p0:{},x_len:{}".format(first_cell, p0, x_len))
#            (fw_g0, bw_g0), _ = bidirectional_dynamic_rnn(first_cell, first_cell,G, x_len, dtype=tf.float32,scope='g0')  # [N, JX, 2d]
            # g0 = tf.concat(3, [fw_g0, bw_g0])
            #self.g0 = tf.concat([fw_g0, bw_g0], 2)
            self.g0,_ = rnn('bi-lstm',  G , x_len, self.hidden_size,dropout_keep_prob=self.dropout_keep_prob)


    def _yesno(self):
        with tf.variable_scope('yesno'):
            batch_size = tf.shape(self.answer)[0]
            concat_passage_encodes = tf.reshape(
                self.g0,
                [batch_size, -1, 2 * self.hidden_size]
            )
            self.yesno1 = tf.layers.dense(tf.reduce_mean(concat_passage_encodes, 1), self.hidden_size, name = 'fc_1' )
            self.yesno1 = tf.nn.dropout(self.yesno1,self.dropout_keep_prob)
            self.yesno2 = tf.layers.dense(self.yesno1, self.hidden_size/10, name = 'fc_2' )
            self.yesno2 = tf.nn.dropout(self.yesno1,self.dropout_keep_prob)
            self.yn_out = tf.layers.dense(self.yesno2, 3, name='fc_3')
           # self.yn_out = tf.nn.sigmoid(self.yn_out)
           # self.yn_out = tf.nn.softmax(self.yn_out)


    def _compute_loss(self):
        """
        The loss function
        """
        if self.classPointMark:  #文章片段选取问题
            labels_one_hot = self.answer
            self.loss = tf.losses.softmax_cross_entropy(labels_one_hot,self.yn_out)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels_one_hot,1),tf.argmax(self.yn_out,1)),tf.float32))
           #self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.yn_out)
           # self.loss = tf.reduce_mean(self.loss)

        self.all_params = tf.trainable_variables()
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        self.learning_rate = cosine_decay_restarts(self.learning_rate, self.global_step, self.num_steps,alpha=self.end_learning_rate)
        self.learning_rate = tf.maximum(self.learning_rate, self.end_learning_rate)
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self,data,batch_size,dropout_keep_prob,epoch,writer):


        bitx = 0


        baseBatch = epoch * (int(len(data) / batch_size))

        total_loss,n_batch_acc = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0


        for  i in range(0, len(data), batch_size):
         #   print(">>>>>>"+str(i)+">>>>>>>>")
            one = data[i:i + batch_size]
            if len(one)<self.batch_size:
                start = random.randint(0,len(data)-self.batch_size-1)
                one=one+data[start:start+self.batch_size-len(one)]
          #  print("will dispose query >>>>>>>>")
            query, query_length,query_mask = padding([x['qWordId'] for x in one], max_len=self.max_q_len)
           # print(">>>>>>q OK >>>")
            passage, passage_length,passage_mask = padding([x['pWordId'] for x in one], max_len=self.max_p_len)
            #print(">>>p OK >>>")
            qChar = padding_char([x['qCharId'] for x in one],pads=0,max_len=self.max_q_len)
           # print (">>>q char OK >>>")
            pChar = padding_char([x['pCharId'] for x in one],pads=0,max_len=self.max_p_len)
           # print (">>>p char OK >>>")
            #qCharMask = padding_char([x['qCharMark'] for x in one],pads=False,max_len=50)
           # pCharMask = padding_char([x['pCharMark'] for x in one],pads=False,max_len=350)
            answer = [x['label'] for x in one]
            #print("====================={}==========================".format(i))
            #print("\n\n\n{}query:{} \nquery_lenghth{}\nquery_mask:{}\npassage:{}\npassage_length:{}\npassage_mask:{}\nqchar:{}\npchar:{}".format(
            #  "*"*20,query, query_length,query_mask,passage, passage_length,passage_mask,qChar,pChar))
            #if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
            #      print ("answer:\n{0}".format(answer))
            feed_dict = {self.p: passage,
                         self.q: query,
                         self.p_length: passage_length,
                         self.q_length: query_length,
                         self.p_mask:passage_mask,
                         self.q_mask:query_mask,
                         self.cp:pChar,
                         self.cq:qChar,
                         self.answer: answer,
                         #self.is_train:True,
                         self.dropout_keep_prob: dropout_keep_prob}
            _, loss ,yn_out,acc= self.sess.run([self.train_op, self.loss,self.yn_out,self.acc], feed_dict)

            bitx += 1
            total_loss += loss
            n_batch_acc += acc
            n_batch_loss += loss


            if log_every_n_batch > 0 and  bitx % log_every_n_batch == 0 :
                summary_loss = tf.Summary(value=[tf.Summary.Value(tag="model/train_loss", simple_value=n_batch_loss/log_every_n_batch), ])
                summary_acc = tf.Summary(value=[tf.Summary.Value(tag="model/train_acc", simple_value=n_batch_acc/log_every_n_batch), ])
                writer.add_summary(summary_loss, i + baseBatch)
                writer.add_summary(summary_acc, i + baseBatch)

            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {},acc is{}'.format(
                    bitx-log_every_n_batch+1, bitx, n_batch_loss / log_every_n_batch,n_batch_acc/log_every_n_batch))
                n_batch_loss = 0
                n_batch_acc = 0

        summary_all_dev_ave_loss = tf.Summary(value=[tf.Summary.Value(tag="model/dev_all_train_loss", simple_value=total_loss/bitx), ])
        writer.add_summary(summary_all_dev_ave_loss, epoch)
        return 1.0 * total_loss / bitx


    def train(self, train_data,dev_data, epochs, batch_size, save_dir, save_prefix,dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """

        writer = tf.summary.FileWriter(self.log_dir)
        max_acc_rate = 0
        for epoch in range(1, epochs + 1):
            trainShuffleData = shuffle_data(train_data,'pWordId' )
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_loss = self._train_epoch(trainShuffleData,batch_size,dropout_keep_prob,epoch,writer)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if dev_data is not None:
                    acc_rate,dev_loss = self.evaluate(dev_data,batch_size,dropout_keep_prob,epoch,writer)
                    self.logger.info('Dev eval accurate {},loss={}'.format(acc_rate,dev_loss))

                    if acc_rate > max_acc_rate:
                        self.save(save_dir, save_prefix)
                        max_acc_rate = acc_rate
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_data, batch_size,dropout_keep_prob,epoch,writer):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        out_acc = 0
        out_loss = 0


        
        log_loss,log_acc=0,0
        log_every_n_batch, n_batch_loss = 10, 0

        bitx = 0
        baseBatch = epoch*(int(len(eval_data)/batch_size))


        for i in range(0, len(eval_data), batch_size):
            one = eval_data[i:i + batch_size]
            if len(one)<self.batch_size:
              continue
              #start = random.randint(0,len(eval_data)-self.batch_size-1)
              #one=one+eval_data[start:start+self.batch_size-len(one)]
            bitx+=1
            query, query_length,query_mask = padding([x['qWordId'] for x in one], max_len=self.max_q_len)
            passage, passage_length,passage_mask = padding([x['pWordId'] for x in one], max_len=self.max_p_len)
            qChar = padding_char([x['qCharId'] for x in one],pads=0,max_len=self.max_q_len)
            pChar = padding_char([x['pCharId'] for x in one],pads=0,max_len=self.max_p_len)
            answer = [x['label'] for x in one]
            feed_dict = {self.p: passage,
                self.q: query,
                self.p_length: passage_length,
                self.q_length: query_length,
                self.p_mask:passage_mask,
                self.q_mask:query_mask,
                self.cp:pChar,
                self.cq:qChar,
                self.answer: answer,
                #self.is_train:True,
                self.dropout_keep_prob: 1.0}
            yn_out , loss,acc = self.sess.run([self.yn_out,self.loss,self.acc], feed_dict)


            out_acc+=acc
            out_loss+=loss
            log_loss+=loss
            log_acc+=acc

            if bitx % log_every_n_batch == 0 :
                summary_loss = tf.Summary(value=[tf.Summary.Value(tag="model/dev_loss", simple_value=log_loss/log_every_n_batch), ])
                summary_acc = tf.Summary(value=[tf.Summary.Value(tag="model/dev_acc", simple_value=log_acc/log_every_n_batch), ])
                writer.add_summary(summary_loss, i + baseBatch)
                writer.add_summary(summary_acc, i + baseBatch)
                log_loss = 0
                log_acc = 0


        summary_all_dev_ave_loss = tf.Summary(value=[tf.Summary.Value(tag="model/dev_all_ave_loss", simple_value=out_loss/bitx), ])
        summary_all_dev_ave_acc = tf.Summary(value=[tf.Summary.Value(tag="model/dev_all_ave_acc", simple_value=out_acc/bitx), ])
        writer.add_summary(summary_all_dev_ave_loss, epoch)
        writer.add_summary(summary_all_dev_ave_acc, epoch)

        return out_acc/bitx , out_loss/bitx

    def get_softmax_result(self, train_data, batch_size, dropout_keep_prob,outputPath):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        out_acc = 0
        out_loss = 0

        log_every_n_batch,n_batch_acc, n_batch_loss = 10, 0,0

        bitx = 0
        fout = open(outputPath,'w')
        for i in range(0, len(train_data), batch_size):
            one = train_data[i:i + batch_size]
            if len(one) < self.batch_size:
                continue
                # start = random.randint(0,len(eval_data)-self.batch_size-1)
                # one=one+eval_data[start:start+self.batch_size-len(one)]
            bitx += 1
            query, query_length, query_mask = padding([x['qWordId'] for x in one], max_len=self.max_q_len)
            passage, passage_length, passage_mask = padding([x['pWordId'] for x in one], max_len=self.max_p_len)
            qChar = padding_char([x['qCharId'] for x in one], pads=0, max_len=self.max_q_len)
            pChar = padding_char([x['pCharId'] for x in one], pads=0, max_len=self.max_p_len)
            queryId = [x['query_id'] for x in one]
            answer = [x['label'] for x in one]
            feed_dict = {self.p: passage,
                         self.q: query,
                         self.p_length: passage_length,
                         self.q_length: query_length,
                         self.p_mask: passage_mask,
                         self.q_mask: query_mask,
                         self.cp: pChar,
                         self.cq: qChar,
                         self.answer: answer,
                         # self.is_train:True,
                         self.dropout_keep_prob: 1.0}
            yn_out, loss, acc = self.sess.run([self.yn_out, self.loss, self.acc], feed_dict)


            out_acc += acc
            out_loss += loss
            n_batch_loss += loss
            n_batch_acc += acc
            for idx,qId  in enumerate(queryId):
                pr = {
                    'query_id':qId,
                    'predict':yn_out[idx].tolist(),
                    'label':answer[idx]
                }
                fout.write("{}\n".format(json.dumps(pr)))

            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {},acc is{}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch, n_batch_acc / log_every_n_batch))
                n_batch_loss = 0
                n_batch_acc = 0

        print("all train data out_acc:{},loss:{}".format(out_acc/bitx,out_loss/bitx))
        return out_acc / bitx, out_loss / bitx
    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
