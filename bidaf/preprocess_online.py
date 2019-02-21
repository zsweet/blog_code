# -*- coding: utf-8 -*-
import pickle as  cPickle
import json
import numpy as np
import jieba
import re
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

def seg_line(line):
#    line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！【】，×X。？：、~@%……&*（）；]", "", line)
    return list(jieba.cut(line))


def seg_data(path):
    print ('start process ', path)
    data = []
    with open(path, 'r',encoding='utf-8') as f:
        for line in f:
            dic = json.loads(line)
            question = dic['query']
            doc = dic['passage']
            alternatives = dic['alternatives']
            data.append([seg_line(question), seg_line(doc), alternatives.split('|'), dic['query_id']])
    return data


def build_word_count(data):
    wordCount = {}

    def add_count(lst):
        for word in lst:
            if word not in wordCount:
                wordCount[word] = 0
            wordCount[word] += 1

    for one in data:
        [add_count(x) for x in one[0:3]]
    print ('word type size ', len(wordCount))
    return wordCount


def build_word2id(wordCount, threshold=10):
    word2id = {'<PAD>': 0, '<UNK>': 1}
    for word in wordCount:
        if wordCount[word] >= threshold:
            if word not in word2id:
                word2id[word] = len(word2id)
        else:
            chars = list(word)
            for char in chars:
                if char not in word2id:
                    word2id[char] = len(word2id)
    print ('processed word size ', len(word2id))
    return word2id






def get_answer_label(seg_query,ansTokens):
    query = ""
    for i in seg_query:
        query+=i
    label = None
    ansTokens = [x.strip() for x in ansTokens]
    unkownMark = False
    unkownIdx = -1
    unkownChar = ['无法确定', '无法确认', '不确定', '不能确定', 'wfqd', '无法选择', '无法确实', '无法取代', '取法确定', '无法确', '无法㾡', '无法去顶', '无确定',
                  '无法去顶', '我放弃', '无法缺定', '无法无额定', '无法判断', '不清楚', '无人确定',"不知道"]

    for idx, token in enumerate(ansTokens):
        for ch in unkownChar:
            if token.find(ch) != -1:
                unkownMark = True
                unkownIdx = idx
                break
        if unkownMark:
            break
            #            print("%s %s %s : %d %s"%(ansTokens[0],ansTokens[1],ansTokens[2],unkownIdx,ansTokens[unkownIdx]))
    minFindStart = 999999
    minIdx = -1
    if unkownMark == False:
        print("%s %s %s unkonwn mark error" % (ansTokens[0], ansTokens[1], ansTokens[2]))
    else:
        for idx, token in enumerate(ansTokens):
            if unkownIdx == idx:
                continue
            tmpFindStart = query.find(token)
            if tmpFindStart == -1:
                tmpFindStart = 999999

            if minFindStart > tmpFindStart:
                minIdx = idx
                minFindStart = tmpFindStart
        if not (minIdx < 0 or minIdx > 2 or unkownMark < 0 or unkownMark > 2):
            if minIdx == 0:
                label = [1,0,0]
            elif unkownIdx == 0 :
                label = [0,0,1]
            else : label = [0,1,0]
        else:
            minIdx = -999
            pessimisticDic = {"不会", "不可以", "不是", "假的", "不要", "不靠谱", "不能", "没有", "不需要", "没出", "不给", "不用", "不可能", "不好", "不同意",
                              "不对", "不算", "不行", "不快", "不能", "没用", "不合适", "不正常", "不好", "不可", "不正确", "不高", "不难", "不属于", "不合适",
                              "不值钱", "不友好", "不幸运", "不应该", "不值"}
            for idx, token in enumerate(ansTokens):
                if idx == unkownIdx:
                    continue
                for opt in pessimisticDic:
                    if token.find(opt) != -1:
                        minIdx = 3 - idx - unkownIdx
            if minIdx != -999:
                if minIdx == 0:
                    label = [1, 0, 0]
                elif unkownIdx == 0:
                    label = [0, 0, 1]
                else:
                    label = [0, 1, 0]
            else:
                minIdx = -999
                for idx, token in enumerate(ansTokens):
                    if token.find("不确定") == -1 and token.find("不能确定") == -1 and (
                                    token.find("不") != -1 or token.find("否") != -1 or token.find(
                                "没") != -1 or token.find("错") != -1):
                        minIdx = 3 - idx - unkownIdx
                if minIdx != -999:
                    if minIdx == 0:
                        label = [1, 0, 0]
                    elif unkownIdx == 0:
                        label = [0, 0, 1]
                    else:
                        label = [0, 1, 0]
                else:
                    pass
                  #  print("after last process ,still failed")
    try:
        if label != None:
            if minIdx == 0:
                if unkownIdx == 1:
                    shuffledAnsToken = [ansTokens[0],ansTokens[2],ansTokens[1]]
                elif unkownIdx ==2:
                    shuffledAnsToken = [ansTokens[0], ansTokens[1], ansTokens[2]]
            elif minIdx == 1:
                if unkownIdx == 0:
                    shuffledAnsToken = [ansTokens[1], ansTokens[2], ansTokens[0]]
                elif unkownIdx == 2:
                    shuffledAnsToken = [ansTokens[1], ansTokens[0], ansTokens[2]]
            elif minIdx == 2:
                if unkownIdx == 0:
                    shuffledAnsToken = [ansTokens[2], ansTokens[1], ansTokens[0]]
                elif unkownIdx == 1:
                    shuffledAnsToken = [ansTokens[2], ansTokens[0], ansTokens[1]]
    except:
        shuffledAnsToken = []
    if shuffledAnsToken == []:
      shuffledAnsToken = ansTokens
    return label,ansTokens,shuffledAnsToken




def transform_data_to_id(raw_data, word2id,char2id,fileOut):
    data = []

    def map_word_to_id(word):
        outputWordId,outputCharId = [],[]
        if word in word2id:
            outputWordId.append(word2id[word])
        else:
            # chars = list(word)
            # for char in chars:
            #     if char in word2id:
            #         outputWordId.append(word2id[char])
            #     else:
                    outputWordId.append(1)
        for ch in word[0:4]:
            if ch in char2id:
                outputCharId.append(char2id[ch])
            else:
                outputCharId.append(1)
        outputCharMark=len(outputCharId)*[True]+(4-len(outputCharId))*[False]
        outputCharId=outputCharId+(4-len(outputCharId))*[0]

        return outputWordId,outputCharId,outputCharMark


    def map_sent_to_id(sent):
        wordIdxList = []
        charIdxList = []
        charMark = []
        #workMark = []
        for word in sent:
            wordIdx,charsIdx,charM = map_word_to_id(word)
            wordIdxList.extend(wordIdx)
            charIdxList.append(charsIdx)
            charMark.append(charM)
        return wordIdxList,charIdxList,charMark


    print("disposing...")
    xxx = 0
    for idx,one in enumerate(raw_data):
        qWordId, pCharId,pWordId,qCharId,label= [],[],[],[],[]
        if word2id is not None:
            qWordId,qCharId,qCharMark= map_sent_to_id(one[0])
            pWordId,pCharId,pCharMark = map_sent_to_id(one[1])
        try:
          label,ansTokens,shuffleRes = get_answer_label(one[0],one[2])
        except:
          xxx += 1
          label = [1,0,0]
          ansTokens = one[2]
          shuffleRes = one[2]
          print("{}:{},{}".format(xxx,one[0],one[1]))
          #continue
        if label == None :
            #data.append([qWordId, pWordId,pCharId,qCharId, one[0],one[2],label, one[2], one[2], one[-1], idx,-1])
            data.append({
                         'qWordId':qWordId,
                         'pWordId':pWordId,
                         'qCharId':qCharId,
                         'pCharId':pCharId,
                         'pCharMark':pCharMark,
                         'qCharMark':qCharMark,
                         'query':one[0],
                         'passage':one[1],
                         'answer':ansTokens,
                         'label':[1,0,0],
                         'shuffleRes':shuffleRes,
                         'query_id':one[-1],
                         'original_dic':one[-2],
                         'successful_mark':1
                        })
            try:
                fileOut.write("query id:{}| successful mark:{} | label:{} | orignal answer:{} "
                              "| shuffleRes:{}| question:{} | doc:{}\n".format(
                                one[-1],-1,label, one[2], one[2],one[0], one[1]))
            except:
                fileOut.write("===print error==={} | successful mark:{}\n".format(one[-1], -1))
        else:
            data.append({
                         'qWordId':qWordId,
                         'pWordId':pWordId,
                         'qCharId':qCharId,
                         'pCharId':pCharId,
                         'pCharMark':pCharMark,
                         'qCharMark':qCharMark,
                         'query':one[0],
                         'passage':one[1],
                         'answer':ansTokens,
                         'label':label,
                         'shuffleRes':shuffleRes,
                         'query_id':one[-1],
                         'original_dic':one[-2],
                         'successful_mark':1
                        })
            try:
                if shuffleRes == []:
                    fileOut.write(
                        "===shuffle error===:query id:{}| successful mark:{} | label:{} | orignal answer:{} "
                        "| shuffleRes:{}| question:{} | doc:{} | question idx:{} | doc idx{} | question_char idx{}  "
                        "| doc_char  idx:{} | qCharMark:{} | pCharMark:{}\n".format(
                            one[-1], 1,label, ansTokens, shuffleRes, one[0], one[1],
                            qWordId, pWordId,qCharId,pCharId,qCharId,qCharMark,pCharMark))
                else:
                  fileOut.write("query id:{}| successful mark:{} | label:{} | orignal answer:{} | shuffleRes:{}"
                                "| question:{} | doc:{} | question idx:{} | doc idx:{} | question_char idx:{}  "
                                "| doc_char idx:{} | qCharMark:{} | pCharMark:{} | original doc:{}\n".format(
                                one[-1], 1,label, ansTokens, shuffleRes, one[0], one[1],
                                qWordId, pWordId,qCharId,pCharId,qCharMark,pCharMark,one[-2]))
            except :
                fileOut.write("===print error==={} | successful mark:{}\n".format(one[-1],1))
            #print(label)
    print("label error number:{}".format(xxx))
    print("data size : "+ str(len(data)))
    return data


def load_pretrained_embeddings_word(embedding_path):
    """
    loads the pretrained embeddings from embedding_path,
    tokens not in pretrained embeddings will be filtered
    Args:
        embedding_path: the path of the pretrained embedding file
    """
    wordList = []
    trainedEmbeddings = {}

    with open(embedding_path, 'r',encoding='utf-8') as file:
        while 1:
            read_line = file.readline()
            if not read_line: break
            line = read_line
            contents = line.strip().split()
            token = contents[0]
            wordList.append(token)
            trainedEmbeddings[token] = list(map(float, contents[1:]))
    return wordList, trainedEmbeddings


def build_word2id_embedding_from_pretrained_embedding(trainedEmbeddings, embed_dim,char_emb_dim):
    word2id = {'<PAD>': 0, '<UNK>': 1}
    char2id = {'<PAD>': 0, '<UNK>': 1}
    for i,word in enumerate(trainedEmbeddings):
        word2id[word.strip()] = len(word2id)
        if len(word.strip())==1:
            char2id[word] = len(char2id)

    #embeddings = np.zeros([len(word2id), embed_dim])
    embeddings = np.random.rand(len(word2id), embed_dim)
    char_emb = np.random.rand(len(char2id),char_emb_dim)
    count = 0
    for i,token in enumerate(word2id):
        if token in trainedEmbeddings:
           # print("==========="+str(i))
            try:
                embeddings[word2id[token.strip()]] = trainedEmbeddings[token.strip()]
                if len(token.strip())==1:
                  char_emb[char2id[token.strip()]] = trainedEmbeddings[token.strip()]
            except Exception as e:
                count+=1
                print(e)
                print ("trainedEmbeddings[token]:"+token+":"+str(len(trainedEmbeddings[token])))
                print (trainedEmbeddings[token])
    print ("aborded embedding number:"+str(count))
    print ("word embedding length:{} char embedding length:{}".format(len(embeddings),len(char_emb)))

    return word2id, embeddings,char2id,char_emb

def process_data(data_path,train_path,valid_path,testa_path, threshold , embed_dim ,char_emb_dim,pretrained_embedding_path ,out_embedding_path , out_char_embedding_path,out_word2id_path,out_char2id_path,versionMark):
   # train_file_path = data_path + 'ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json'
    #train_file_path = data_path + train_path
    dev_file_path = valid_path
    test_a_file_path =  testa_path
    path_lst = [dev_file_path, test_a_file_path]
    #path_lst = [train_file_path]

    output_path = [data_path + x for x in ['dev'+versionMark+'.pickle', 'testa'+versionMark+'.pickle']]
    #output_path = [data_path + x for x in ['train.pickle']]
    return _process_data(path_lst, output_path ,embed_dim ,char_emb_dim,out_embedding_path,out_char_embedding_path,out_word2id_path,out_char2id_path,
                         pretrained_embedding_path,loading_pretrained_embedding=1,word_min_count=threshold,versionMark=versionMark)



###input file list , threshold , output file list
def _process_data(path_lst,output_file_path,embed_dim,char_emb_dim,out_embedding_path,out_char_embedding_path,out_word2id_path, out_char2id_path,pretrained_embedding_path = "",loading_pretrained_embedding = 0 ,word_min_count=5,versionMark = ""):
    raw_data = []
    #对原始语料分词  注意答案的分词方式是通过语料中的 | 直接分词的
    for path in path_lst:
        raw_data.append(seg_data(path))
    word2id, char2id = None, None
    print("seg is OK...")
    if loading_pretrained_embedding==1:
        # wordList,trainedEmbeddings = load_pretrained_embeddings_word(pretrained_embedding_path)
        #
        # word2id,embedding,char2id,char_embedding = build_word2id_embedding_from_pretrained_embedding(trainedEmbeddings,embed_dim,char_emb_dim)
        # print("embedding length:" + str(len(embedding)) + "word2id:" + str(len(word2id)))
        # with open(out_embedding_path, 'wb') as f:
        #     cPickle.dump(embedding, f)
        # print ("write word2id.obj to data/word2id.obj...")
        # with open(out_word2id_path, 'wb') as f:
        #     cPickle.dump(word2id, f)
        #
        # print("char embedding length:" + str(len(char_embedding)) +"*"+str(len(char_embedding[0])) + "char2id:" + str(len(char2id)))
        # with open(out_char_embedding_path, 'wb') as f:
        #         cPickle.dump(char_embedding, f)
        # print("write char2id.obj to data/word2id.obj...")
        # with open(out_char2id_path, 'wb') as f:
        #             cPickle.dump(char2id, f)

        with open(out_char2id_path, 'rb') as f:
            char2id = cPickle.load(f)

        with open(out_word2id_path, 'rb') as f:
            word2id = cPickle.load(f)



        #print("write words.table to data/words.table...")
        # with open('data/words.table', 'w') as f:
        #     s = ""
        #     for word in trainedEmbeddings:
        #         s="{0}:{1}\n".format(word.decode('gbk').encode('utf-8'),trainedEmbeddings[word])
        #         f.write(s)



    elif loading_pretrained_embedding == 0:
        word_count = build_word_count(
            [y for x in raw_data for y in x])  # [[question,doc,answer,id],[],[]...]  word_count = {"":}
        with open('data/word-count.obj', 'wb') as f:
            cPickle.dump(word_count, f)
        word2id = build_word2id(word_count, word_min_count)  # 对分词之后的建立所有，小于threshold的词将被分成char，然后加入词表
        with open('data/word2id.obj', 'wb') as f:
            cPickle.dump(word2id, f)
    print("dispose data and write...")
    for one_raw_data, one_output_file_path in zip(raw_data, output_file_path):
        with open(one_output_file_path, 'wb') as f:
            with open(one_output_file_path+"_log.txt", 'w') as  logFout:
                one_data = transform_data_to_id(one_raw_data, word2id,char2id,logFout)
                cPickle.dump(one_data, f)
    return len(word2id)
