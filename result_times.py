# -*- coding: utf-8 -*-
import collections
import re

class result_times():
    def __init__(self,file_path):
        with open(file_path) as f:
            self.text = f.read()
            self.text = re.sub("[ \n]","",self.text)
            
    
    def split(self):
        list = re.split("[|]",self.text)
        self.calc_result(list)
    
    def calc_result(self,list):
        result = collections.Counter(list).most_common()
        print(result)
        self.output(result)

    def output(self,result):
        with open("result_times.csv","w") as f:
            for word in result:
                f.write(str(word[0])+","+str(word[1])+"\n")

if __name__ == "__main__":
    rt = result_times("result/result.txt")
    rt.split()