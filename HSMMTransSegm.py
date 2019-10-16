#-*- coding: utf-8 -*-
#from __future__ import unicode_literals
import numpy as np
import random
import math
import time
import codecs
import os


class HSMMWordSegm():
    # 事前分布
    MAX_LEN = 15 # 考えられる最大の文節の個数　（多すぎると学習の回数が多くなる→データに基づき妥当な値にする）
    AVE_LEN = 3 # 平均的な分節の個数(多すぎるとうまく分割されない）

    def __init__(self, nclass ):
        self.num_class = nclass
        self.segm_class = {}
        self.segm_sequences = []
        self.trans_prob = np.ones( (nclass,nclass) )
        self.trans_prob_bos = np.ones( nclass )
        self.trans_prob_eos = np.ones( nclass )

        
    # クラスcに分節を追加
    def add_segm(self, c, segm ):
        # 引数入力例: 0,[1,2,1](segmはデータをランダムに分割したもの)
        # segmのIDとランダムに割り振った数字を辞書化
        # {segmのID:c(ランダム),segmのID:c(ランダム)...}
        self.segm_class[id(segm)] = c

        #print("================")
        #print(segm)
        for i in range(len(segm)-1):
            # segm内(例: [5,5,5,4,8...])の文字列内の遷移回数
            o1 = segm[i]
            o2 = segm[i+1]
            self.obs_trans_count[c][o1][o2] += 1 # n番目の数からn+1番目の数が続いた回数
            self.obs_count[c][o1] += 1  # 数字が出た回数

            #print("{}→{}".format(o1,o2))
            #print(self.obs_trans_count[c][o1][o2])
        
    # クラスcから分節を削除
    def delete_segm(self, c, segm):
        # N人目のデータ(分割済)を1単語ずつポップで取り出すことで分節を削除
        self.segm_class.pop(id(segm))
        for i in range(len(segm)-1):
            o1 = segm[i]
            o2 = segm[i+1]
            self.obs_trans_count[c][o1][o2] -= 1
            self.obs_count[c][o1] -= 1
            

    def load_data(self, filename ):
        # 観測シーケンス
        # データセットの各行をそれぞれリストにする [[1行目],[2行目],[3行目]]
        self.sequences = [ [ int(i) for i in line.split() ] for line in open( filename ).readlines()]
        
        # 観測の種類
        #print( self.sequences )
        # リストに格納された値の最大値+1をnum_obsに代入(現在は1～10まで割り振っているので11が入る)
        self.num_obs = int(np.max( [ np.max(s) for s in self.sequences] )+1) 

        
        # 観測の遷移確率を計算するためのパラメータ
        self.obs_trans_count = np.zeros((self.num_class,self.num_obs,self.num_obs) ) # クラス数,状態数+1,状態数+1
        self.obs_count = np.zeros( (self.num_class,self.num_obs) ) # クラス数,状態数+1

        # ランダムに分節化        
        self.segm_sequences = []
        for seq in self.sequences:
            segms = []

            i = 0
            while i<len(seq): #（1行ごとに）i以上であれば繰り返す
                # ランダムに切る(切るところは，1～MAX_LENの乱数で決定)
                length = random.randint(1,self.MAX_LEN)

                if i+length>=len(seq):
                    length = len(seq)-i

                segms.append( seq[i:i+length] ) # 分節した箇所でsegmsに格納
                i+=length # 分節後の先頭を変更するための処理

            self.segm_sequences.append( segms ) # 1行分のランダム分節結果をsegm_sequenceに格納

            # ランダムに割り振る (変数iの意味は？)
            for i,segm in enumerate(segms): # 例: 0,[1,2,1]
                c = random.randint(0,self.num_class-1) # 0～データ数-1の値をランダムに代入
                self.add_segm( c, segm ) # ランダムに分割したsegmをadd_segmに渡す

        # 遷移確率更新
        self.calc_trans_prob()

    def calc_output_prob(self, c , segm ):
        # 長さの制約
        L = len(segm)
        print(L)
        prior = (self.AVE_LEN**L) * math.exp( -self.AVE_LEN ) / math.factorial(L)
        
        # クラスcの遷移確率から生成される確率
        p = prior
        for i in range(len(segm)-1):
            o1 = segm[i]
            o2 = segm[i+1]
            p *= (self.obs_trans_count[c][o1][o2] + 0.1)/(self.obs_count[c][o1] + self.num_obs*0.1 )
        
        return p

    def forward_filtering(self, sequence):
        print("===========")
        print(sequence)
        T = len(sequence)
        a = np.zeros( (len(sequence), self.MAX_LEN, self.num_class) ) # 前向き確率(N行目の文字数*15*10)
        for t in range(T):
            for k in range(self.MAX_LEN):  
                if t-k<0:
                    break

                for c in range(self.num_class):
                    """
                    クラス{0..9},文字列{N行目n文字目..N行目n+15-1文字目}の総当たりをcalc_output_probに投げる 
                    例: 10クラス文字列[1,2,3]だった場合
                    c0[1],c1[1]...c9[1]
                    c0[2],c1[2]...c9[2] / c0[2,1],c1[2,1]...c9[2,1]
                    c0[3],c1[3]...c9[3] / c0[2,3],c1[2,3]...c9[2,3] / c0[3,2,1],c1[3,2,1]...c9[3,2,1]
                    """
                    #print("c: {},seq{}".format(c,sequence[t-k:t+1]))
                    out_prob = self.calc_output_prob( c , sequence[t-k:t+1] )

                    # 遷移確率
                    tt = t-k-1 # 文頭だけ別扱いするための計算
                    if tt >= 0:
                        # ベイズ推論による機械学習入門　p.189の5.122式に対応？
                        for kk in range(self.MAX_LEN):
                            for cc in range(self.num_class):
                                a[t, k, c] += a[tt, kk, cc] * self.trans_prob[cc, c]
                        a[t,k,c] *= out_prob

                    else:
                        # 最初の単語
                        a[t,k,c] = out_prob * self.trans_prob_bos[c]
                        

                    # 最後の単語の場合
                    if t==T-1:
                        a[t,k,c] *= self.trans_prob_eos[c]   
        return a

    def sample_idx(self, prob ):
        accm_prob = [0,] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i

    def backward_sampling(self, a, sequence, use_max_path=False):
        T = a.shape[0]
        t = T-1

        segms = []
        classes = []

        c = -1
        while True:

            # 状態cへ遷移する確率
            if c==-1:
                trans = np.ones( self.num_class )
            else:
                trans = self.trans_prob[:,c]

            if use_max_path:
                idx = np.argmax( (a[t]*trans).reshape( self.MAX_LEN*self.num_class ) )
            else:
                idx = self.sample_idx( (a[t]*trans).reshape( self.MAX_LEN*self.num_class ) )


            k = int(idx/self.num_class)
            c = idx % self.num_class

            segm = sequence[t-k:t+1]

            segms.insert( 0, segm )
            classes.insert( 0, c )

            t = t-k-1

            if t<0:
                break

        return segms, classes


    def calc_trans_prob( self ):
        self.trans_prob = np.ones( (self.num_class,self.num_class) ) # 10*10 リスト
        self.trans_prob_bos = np.ones( self.num_class )# 1*10 リスト
        self.trans_prob_eos = np.ones( self.num_class ) # 1*10 リスト

        # 数え上げる
        #print("====")
        for n,segms in enumerate(self.segm_sequences):
            try:
                # BOS(文頭)　出現頻度？
                c = self.segm_class[ id(segms[0]) ]
                self.trans_prob_bos[c] += 1

            except KeyError as e:
                # gibss samplingで除かれているものは無視
                continue

            for i in range(1,len(segms)):
                # 途中の遷移　遷移頻度
                cc = self.segm_class[ id(segms[i-1]) ]
                c = self.segm_class[ id(segms[i]) ]
                self.trans_prob[cc,c] += 1.0

                # 以下のコメントアウトを外すと，遷移の推移（回数）が見れる
                #print(cc,c)
                #print(self.trans_prob)

            # EOS(文末) 出現頻度？
            c = self.segm_class[ id(segms[-1]) ]
            self.trans_prob_eos[c] += 1
         
        # 正規化(回数から確率を算出)
        # 文頭・文末以外: クラスnからn+1に遷移した回数から確率をそれぞれ割り出す (クラス数より10*10の遷移確率表が出来る)
        self.trans_prob = self.trans_prob / self.trans_prob.sum(1).reshape(self.num_class,1)

        # 文頭のみ: クラスnからn+1に遷移した回数から確率をそれぞれ割り出す (クラス数より1*10の遷移確率表が出来る)
        self.trans_prob_bos = self.trans_prob_bos / self.trans_prob_bos.sum()

        # 文末のみ: クラスnからn+1に遷移した回数から確率をそれぞれ割り出す (クラス数より1*10の遷移確率表が出来る)
        self.trans_prob_eos = self.trans_prob_eos / self.trans_prob_eos.sum()


    def print_result(self):
        temp_number = 1
        print ("-------------------------------")
        # 1行ずつ(Noごとに)処理
        for segms in self.segm_sequences: 
            print("No."+str(temp_number))
            # 1行(あるNoが)分割された結果のリストと割り当てられたクラスの出力
            for s in segms: 
                print( s, ":", self.segm_class[id(s)] )
            temp_number += 1
            print ("------------")
        print("")

        for c in range(self.num_class):
            print( "class", c )
            print( self.obs_trans_count[c] )
            print()


    def learn(self,use_max_path=False):
        for i in range(len(self.sequences)):
            seq = self.sequences[i] #N行目(N人目)の未分割データ
            segms = self.segm_sequences[i] #N行目(N人目)分割済データ
            
            # 学習データから削除
            for s in segms:
                c = self.segm_class[id(s)]
                self.delete_segm(c, s)


            # 遷移確率更新
            self.calc_trans_prob()

            # foward確率計算
            a = self.forward_filtering( seq )

            # backward sampling
            # segms: 分節化されたシーケンス
            # classes: 各分節が分類されたクラス
            segms, classes = self.backward_sampling( a, seq, use_max_path )

            # パラメータ更新
            self.segm_sequences[i] = segms
            for s,c in zip( segms, classes ):
                self.add_segm( c, s )

            # 遷移確率更新
            self.calc_trans_prob()
        return
    
    # 対数尤度（Log likelihood）の計算
    def calc_loglik(self):
        lik = 0
        for segms in self.segm_sequences:
            for i in range(len(segms)-1):
                s1 = segms[0]
                s2 = segms[1]
                
                c1 = self.segm_class[id(s1)]
                c2 = self.segm_class[id(s2)]
                
                lik += math.log( self.calc_output_prob(c1, s1) * self.trans_prob[c1,c2] )
            # BOS
            c1 = self.segm_class[id(segms[0])]
            lik += math.log( self.trans_prob_bos[c1] )
            
            # EOS
            s1 = segms[-1]
            c1 = self.segm_class[id(s1)]
            
            lik += math.log( self.calc_output_prob(c1, s1) * self.trans_prob_eos[c1] )

        return lik            

    def save_result(self, dir ):
        if not os.path.exists(dir):
            os.mkdir(dir)

        for c in range(self.num_class):
            fname = os.path.join(dir, "trans_count_%03d.txt" % c)
            # print(self.obs_trans_count[c])
            np.savetxt( fname, self.obs_trans_count[c] )

        path = os.path.join( dir , "result.txt" )
        f = codecs.open( path ,  "w" , "sjis" )

        for segms in self.segm_sequences:
            for s in segms:
                for o in s:
                    f.write( o )
                f.write( " | " )
            f.write("\n")
        f.close()

        np.savetxt( os.path.join(dir,"trans.txt") , self.trans_prob , delimiter="\t" )
        np.savetxt( os.path.join(dir,"trans_bos.txt") , self.trans_prob_bos , delimiter="\t" )
        np.savetxt( os.path.join(dir,"trans_eos.txt") , self.trans_prob_eos , delimiter="\t" )


def main():
    segm = HSMMWordSegm( 10 ) # データの個数を指定
    segm.load_data( "playcommand.txt" )
    segm.print_result()

    for it in range(50):
        print( it )
        segm.learn()
        print( "lik =", segm.calc_loglik() )

    segm.learn( True )
    segm.save_result("result")
    segm.print_result()
    return


if __name__ == '__main__':
    main()