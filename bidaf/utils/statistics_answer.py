import sys
import json

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def get_antitude():
    if (query(ansTokens[0]))

if __name__ == "__main__":
    dic = {}
    fileList = ["./ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json",
                "./ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json"]
    count = 0
    for path in fileList:
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                count+=1
                if count%1000==0:
                    print (count)
                ttt = json.loads(line, encoding='utf-8')
                query = ttt['query']
                alternatives = ttt['alternatives']
                ansTokens = alternatives.split("|")
                print ("%s        %s | %s | %s" % (query,ansTokens[0],ansTokens[1],ansTokens[2]))

                for token in ansTokens:
             #     print (token)
                  if token in dic.keys():
                        dic[token]+=1
                  else:
                        dic[token]=1
            break
    #print("len:"+str(len(dic)))
    #for d in dic:
    #  print("{0} : {1}".format(d,dic[d]))
    #print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n\n\n\n"
    res = sorted(dic.items(),key=lambda x:x[1],reverse=True)
    for r in res:
        print ("{0} : {1}".format(r[0],r[1]))

