import math
import sys
from pprint import pprint
from calibrate import calibrate
import pandas as pd


def cal_weights(weights):
    r = []
    sum = 0.0
    for w in weights:
        sum += w
    for w in weights:
        r.append(w/sum)
    return r
    
def get_pred_ctr(inp):
    return pd.read_csv(inp).proba.values

def ensemble(weights,files,output):
    ctrs = []
    weights = cal_weights(weights)
    f = open(output,"w")
    for file in files:
        ctr = get_pred_ctr(file)
        print("loading "  + file)
        ctrs.append(ctr)
    sample_num = len(ctrs[0])

    for j in range(sample_num):
        cur_ctr = 0.0
        for k in range(len(ctrs)):
            cur_ctr += weights[k] * math.log(ctrs[k][j]/(1-ctrs[k][j]))
        cur_ctr = 1/(1+math.exp(-cur_ctr))
        print(str(cur_ctr), file=f)
    f.close()

def sub(result,testfile,output):
    f = open(testfile)
    r = open(result)
    o = open(output,"w")
    l1 = f.readline()
    print("instanceID,proba", file=o)
    while True:
        l1 = f.readline()
        l2 = r.readline().strip()
        if not l1:
            break
        print(l1.split(',')[0]+","+l2, file=o)
    f.close()
    r.close()
    o.close()

# files = ["../ftrl_1","../ftrl_2","../fm_test_2.out","../fm_test_2_split"]

# test only (Jie)
files = sys.argv[1:]

weights = [1] * len(files)

ensemble(weights,files,"_ensemble")
sub("_ensemble","data/pre/test.csv","_ensemble_sub")
calibrate("_ensemble_sub","ensemble_cal.csv")
