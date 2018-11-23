# ==============================================================================
# Author: sxron
# E-mail: shixiang08abc@gmail.com
# Copyright 2017 Sogou Inc. All Rights Reserved.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import numpy as np
import tensorflow as tf

from tensorflow.core.protobuf import saver_pb2

class Word2Vec(object):
  def __init__(self,config):
      self.word2id = {}
      self.id2word = []
      self.num_steps = config.max_length
      self.batch_size = config.batch_size
      self.vocab_size = config.vocab_size
      self.target_delay = config.target_delay
      self.embedding_dim = config.embedding_dim
      #self.tag_size = config.tag_size
      #self.tag_dim = config.tag_dim
      #with tf.device("cpu:0"):
      self.id2embedding = None#np.random.rand([self.vocab_size+1,self.embedding_dim])#tf.get_variable("embedding",[self.vocab_size+1,self.embedding_dim],dtype=tf.float32,trainable=True)
        #self.id2tagembedding = tf.get_variable("tag_embedding",[self.tag_size+1,self.tag_dim],dtype=tf.float32,trainable=True)

  def loadWordFile(self,filename):
      sys.stderr.write("\nloading word table...")
      fin = open(filename,'r')
      while True:
          word = fin.readline()
          word = word.strip()
          if word=="":
            break
          self.word2id[word] = len(self.id2word)
          self.id2word.append(word)
      fin.close()
      sys.stderr.write("\nloading word table finished!!!")

  def loadEmbedding(self,session,filepath):
      sys.stderr.write("\nloading word embeddings form tensor file...")
      saver = tf.train.Saver({'embedding':self.id2embedding})
      saver.restore(session,filepath)
      sys.stderr.write("\nloading word embeddings finished !!!")

  # def loadTagEmbedding(self,session,filepath):
  #     sys.stderr.write("\nloading tag embeddings form tensor file...")
  #     saver = tf.train.Saver({'tag_embedding':self.id2tagembedding})
  #     saver.restore(session,filepath)
  #     sys.stderr.write("\nloading tag embeddings finished !!!")

  def getWid(self,term):
      if self.word2id.has_key(term):
        return self.word2id[term]
      else:
        return self.vocab_size

  def getWord(self,wid):
      assert wid<=self.vocab_size
      return self.id2word[wid]

  # def getTagIdx(self, hexStr):
  #     binStr = bin(int(hexStr,16))[2:]
  #     idx = binStr.find('1')
  #     idx = len(binStr) - idx
  #     #if (idx>len(binStr)) or (idx<0) or (idx>self.tag_size):
  #     if (idx > len(binStr)) or (idx < 0):
  #       idx = 0
  #     return idx

  def getImportanceBatchData(self, fin,discardLines):
      batch_cursor = 0
      data_list = np.zeros([self.batch_size, self.num_steps+self.target_delay], dtype=np.int32)
      #tag_list = np.zeros([self.batch_size, self.num_steps+self.target_delay], dtype=np.int32)
      target_list = np.zeros([self.batch_size, self.num_steps+self.target_delay], dtype=np.float32)
      length_list = np.zeros([self.batch_size], dtype=np.int32)
      frame_weight = np.zeros([self.batch_size, self.num_steps+self.target_delay], dtype=np.float32)

      while True:
          line = fin.readline()
          if line=="":
              if batch_cursor==0:
                   # return False,data_list,tag_list,target_list,length_list,frame_weight
                   return False, data_list, target_list, length_list, frame_weight,discardLines
              else:
                  # return True,data_list,tag_list,target_list,length_list,frame_weight
                  return True, data_list, target_list, length_list, frame_weight,discardLines

          line = line.strip()
          tokens = line.split('\t')
          # if len(tokens) != 2:
          if len(tokens) != 2:
              print ("Invalid line: %s , must have 2 fields." % line)
              continue
          terms = tokens[0].strip().split(" ")
          labels = tokens[1].strip().split(" ")
          #tags = tokens[2].strip().split(" ")
          # if len(terms)!=len(labels) or len(terms)!=len(tags):
          if len(terms) != len(labels):
              print ("Invalid line: %s , different length between sentence and target." % line)
              continue
          ###????????????????????????
          if len(terms)>self.num_steps:
              #print ("discard line: %s , this sentence length longer than num_steps." % line)
              discardLines+=1
              continue

          isFalse = False
          for i in range(len(terms)):
              try:
                  data_list[batch_cursor, i] = self.getWid(terms[i].strip())
                  #tag_list[batch_cursor, i] = self.getTagIdx(tags[i].strip())
                  target_list[batch_cursor, i+self.target_delay] = float(labels[i])
                  ###can modify the parameter!!!!!!!
                  if float(labels[i])>0.9 or float(labels[i])<0.2:
                    frame_weight[batch_cursor, i+self.target_delay] = 2.0
                  else:
                    frame_weight[batch_cursor, i+self.target_delay] = 1.0
              except:
                  #print ("discard line: %s , this sentence value error." % line)
                  discardLines+=1
                  isFalse = True
                  break
          length_list[batch_cursor] = len(terms) + self.target_delay
          if isFalse:
              continue

          batch_cursor += 1
          if batch_cursor==self.batch_size:
              #return True,data_list,tag_list,target_list,length_list,frame_weight
              return True, data_list, target_list, length_list, frame_weight,discardLines
  def getImportanceBatchDataFromOnline(self, fin,discardLines):
      batch_cursor = 0
      data_list = np.zeros([self.batch_size, self.num_steps+self.target_delay], dtype=np.int32)
      #tag_list = np.zeros([self.batch_size, self.num_steps+self.target_delay], dtype=np.int32)
      target_list = np.zeros([self.batch_size, self.num_steps+self.target_delay], dtype=np.float32)
      length_list = np.zeros([self.batch_size], dtype=np.int32)
      frame_weight = np.zeros([self.batch_size, self.num_steps+self.target_delay], dtype=np.float32)

      while True:
          line = fin.readline()
          if line=="":
              if batch_cursor==0:
                   # return False,data_list,tag_list,target_list,length_list,frame_weight
                   return False, data_list, target_list, length_list, frame_weight,discardLines
              else:
                  # return True,data_list,tag_list,target_list,length_list,frame_weight
                  return True, data_list, target_list, length_list, frame_weight,discardLines

          line = line.strip()
          tokens = line.split('\t')
          # if len(tokens) != 2:
          if len(tokens) != 2:
              print ("Invalid line: %s , must have 2 fields." % line)
              continue
          terms = tokens[0].strip().split(" ")
          labels = tokens[1].strip().split(" ")
          #tags = tokens[2].strip().split(" ")
          # if len(terms)!=len(labels) or len(terms)!=len(tags):
          if len(terms) != len(labels):
              print ("Invalid line: %s , different length between sentence and target." % line)
              continue
          ###????????????????????????
          if len(terms)>self.num_steps:
              #print ("discard line: %s , this sentence length longer than num_steps." % line)
              discardLines+=1
              continue

          isFalse = False
          for i in range(len(terms)):
              try:
                  data_list[batch_cursor, i] = self.getWid(terms[i].strip())
                  #tag_list[batch_cursor, i] = self.getTagIdx(tags[i].strip())
                  target_list[batch_cursor, i+self.target_delay] = float(labels[i])
                  ###can modify the parameter!!!!!!!
                  if float(labels[i])>0.9 or float(labels[i])<0.2:
                    frame_weight[batch_cursor, i+self.target_delay] = 2.0
                  else:
                    frame_weight[batch_cursor, i+self.target_delay] = 1.0
              except:
                  #print ("discard line: %s , this sentence value error." % line)
                  discardLines+=1
                  isFalse = True
                  break
          length_list[batch_cursor] = len(terms) + self.target_delay
          if isFalse:
              continue

          batch_cursor += 1
          if batch_cursor==self.batch_size:
              #return True,data_list,tag_list,target_list,length_list,frame_weight
              return True, data_list, target_list, length_list, frame_weight,discardLines

if __name__=="__main__":
  pass
