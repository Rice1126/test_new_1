import matplotlib.pyplot as plt
import random
import numpy as np
import time
import math



plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
H = [1,2]
C = [12,13]
N = [14,15]
O = [16,17,18]
Na = [23]
K = [39,40,41]
Cs = [133]
Hf = [174,176,177,178,179,180]



H_P = [999,1]
C_P = [99,1]
N_P = [996,4]
O_P = [9976,4,20]
K_P = [9326,1,673]
Hf_P = [16,526,1860,2728,1362,3508]


Hf_arr = [[174,176,177,178,179,180],
          [16,526,1860,2728,1362,3508], ]
H_arr = [[1,2],[999,1] ]
C_arr = [[12,13],[99,1]]
N_arr = [[14,15],[996,4]]
O_arr = [ [16,17,18], [9976,4,20]]

Na_arr = [[23],[1]]

K_arr = [[39,40,41],[9326,1,673]]

Cs_arr = [[133],[1]]



EleList = {
    "Hf": Hf_arr,
    "O": O_arr,
    "H":H_arr,
    "C":C_arr,
    "N":N_arr,
    "Na":Na_arr,
    "K":K_arr,
    "Cs":Cs_arr
    }

CompList = {
    "Hf12O22H14":{"Hf":12, "O":22, "H":14},
    "H2O":{"H":2, "O":1},
    "OH-":{"H":1, "O":1},
    "H7C3ON":{"H":7, "C":3, "O":1, "N":1},
    "HCO2-":{"H":1, "C":1, "O":2, "C":1},
    "HCO2H":{"H":2, "C":1, "O":2, "C":1},
    "H+":{"H":1},
    "Na+":{"Na":1},
    "K+":{"K":1},
    "Cs+":{"Cs":1},
    }

Combine1 = {
    "Hf12O22H14":1,
    "H2O":0,
    "OH-":0,
    "H7C3ON":0,
    "HCO2-":18,
    "HCO2H":0,
    "H+":1,
    "Na+":0,
    "K+":0,
    "Cs+":0
    }


Combine2 = {
    "Hf12O22H14":0.994,
    "H2O":1,
    "OH-":1,
    "H7C3ON":0,
    "HCO2-":17,
    "HCO2H":0,
    "H+":1,
    "Na+":0,
    "K+":0,
    "Cs+":0
    }

Combine3 = {
    "Hf12O22H14":1.0,
    "H2O":0,
    "OH-":1,
    "H7C3ON":1,
    "HCO2-":17,
    "HCO2H":0,
    "H+":1,
    "Na+":0,
    "K+":0,
    "Cs+":0
    }



def getMean(Ele):
    wei = Ele[0]
    per = Ele[1]

    s = 0
    for i in per:
        s = i+s
    

    s1 = 0.0
    s2 = 0.0
    for i in range(0, len(per)):
        s1 = s1 + per[i]*wei[i]
        s2 = s2 + per[i]
    return s1/s2


def getVar(Ele):
    wei = Ele[0]
    per = Ele[1]

    s1 = 0.0
    s2 = 0.0
    for i in range(0, len(per)):
        s1 = s1 + per[i]*wei[i]*wei[i]
        s2 = s2 + per[i]

    mean_sqr = s1/s2

    
    var = mean_sqr - pow(getMean(Ele),2)
    return var


def compVar(name):
    comp = CompList[name] 
    s = 0
    for key in comp.keys():
        ele_arry = EleList[key]  
        num = comp[key]
        
        s = s + num*getVar(ele_arry)

    return s   



def compMean(name):
    comp = CompList[name] 
    s = 0
    for key in comp.keys():
        ele_arry = EleList[key]  
        num = comp[key]
        
        s = s + num*getMean(ele_arry)

    return s   
    
        
def getRandomIndex(dist):
    
    total = []
    t = 0
    for x in dist:
        t = t + x
        total.append(t)
        
    p = random.random()

    p_i = p * total[-1]
    for i in range(0, len(dist)):
        if p_i < total[i]:
            return i
    return len(dist) - 1

def getComp(name):
    comp = CompList[name]
    res = 0
    for key in comp.keys():
        ele_arr = EleList[key]
        num = comp[key]
        for i in range(num):
            gi = getRandomIndex(ele_arr[1])
            ato = ele_arr[0][gi]
            res = res + ato
    return res



def getCompListWeight(combine):
    

    
    value = 0
  
    for key in combine.keys():        
 
        wei = combine[key]
        mol = getComp(key)
        value = value + mol*wei


    return value


def getGaussionParameter(combine):

    mu = 0
    var = 0
    for key in combine.keys():
        mu = mu + compMean(key)*combine[key]
        var = var + compVar(key)*combine[key]
    sigma = pow(var,0.5)

    return mu,sigma

def getDataByCombine(combine,lengths):
    
    Y = []
    
    for i in range(lengths):       
        val = getCompListWeight(combine)
        
        Y.append(val)
        
    return Y

    

    
def loadData(combine_list, datalengths_list):

    mu = []
    sigma = []

    
    for comb in combine_list:
        m,s = getGaussionParameter(comb)
        mu.append(m)
        sigma.append(s)

    #打印出理论的均值和标准差
    for i in range(len(mu)):
        print("muxx,sigmaxx_____:",mu[i],sigma[i])

    
    
    dataSet = []
    for i in range(len(combine_list)):
        Y = getDataByCombine(combine_list[i],datalengths_list[i])
        dataSet = dataSet + Y

    return dataSet, mu, sigma
    


colors = ["red","blue","green","brown","yellow","gold","gray"]
signs = [".",".","*","*","o"]

def plotData(dataSet,datalengths_list):

 
    j = 0
    ci = 0
    for len0 in datalengths_list:
        Y = []
        for x in range(len0):
            Y.append(dataSet[j])
            j = j+1
                
        d = {}
        for y in Y:

            if y in d.keys():
                d[y] = d[y]+ 1
            else:
                d[y] = 1
        x = list(d.keys())
        y = list(d.values())


        xp = x[:]
        xp.sort()


        plt.plot(x,y,signs[ci],color = colors[ci])
        ci = ci+1


    dd ={}
    for yy in dataSet:
        if yy in dd.keys():
                dd[yy] = dd[yy]+ 1
        else:
                dd[yy] = 1

    xx = list(dd.keys())
    yy = list(dd.values())
    plt.plot(xx,yy,signs[ci],color = colors[ci])
        
    plt.show()




def calcGauss(dataSetArr, mu, sigmod):


    result = (1 / (math.sqrt(2 * math.pi) * sigmod)) * \
             np.exp(-1 * (dataSetArr - mu) * (dataSetArr - mu) / (2 * sigmod**2))

    return result


def E_step(dataSetArr, mu_list, sigma_list, alpha_list):



    gamma = []
    sumx = 0
    for i in range(len(mu_list)):
        ga = alpha_list[i] * calcGauss(dataSetArr, mu_list[i], sigma_list[i] )
        sumx = sumx + ga
        gamma.append(ga)

    gamma_new = []
    for ga in gamma:
        ga_new = ga/sumx
        gamma_new.append(ga_new)

    return gamma_new
        


def M_step(dataSetArr, mu_list, gamma_list):



    mu_list_new = []
    sigma_list_new = []
    alpha_list_new = []
    for i in range(len(mu_list)):
        mu_new = np.dot(gamma_list[i], dataSetArr) / np.sum(gamma_list[i])
        mu_list_new.append(mu_new)

        sigma_new = math.sqrt(np.dot(gamma_list[i], (dataSetArr - mu_list[i])**2) / np.sum(gamma_list[i]))
        sigma_list_new.append(sigma_new)

        alpha_new = np.sum(gamma_list[i]) / len(gamma_list[i])
        alpha_list_new.append(alpha_new)


    return mu_list_new, sigma_list_new, alpha_list_new
        


def EM_Train(dataSetList,mu_list, sigma_list, alpha_list, iter = 200):
    '''
    根据EM算法进行参数估计
    :param dataSetList:数据集（可观测数据）
    :param iter: 迭代次数
    :return: 估计的参数
    '''

    print("对参数取初值alpha0: ", alpha_list)
    #将可观测数据y转换为数组形式，主要是为了方便后续运算
    dataSetArr = np.array(dataSetList)

    #步骤1：对参数取初值，开始迭代
    #alpha0 = 0.33; mu0 = 3248.05140; sigmod0 = 4.556270417786535
    #alpha1 = 0.33; mu1 = 3322.1627320; sigmod1 = 4.581614501024358 
    #alpha2 = 0.34; mu2 = 3216.951868; sigmod2 = 4.53297309544152

 


    #开始迭代
    step = 0


    while (step < iter):
        #每次进入一次迭代后迭代次数加1
        step += 1
        
        #步骤2：E步：依据当前模型参数，计算分模型k对观测数据y的响应度

 
        gamma_list = E_step(dataSetArr, mu_list, sigma_list, alpha_list)
        #步骤3：M步

    
        mu_list, sigma_list, alpha_list =   M_step(dataSetArr, mu_list, gamma_list)


    #迭代结束后将更新后的各参数返回

    return alpha_list, mu_list, sigma_list


if __name__ == '__main__':
    start = time.time()
    


    #初始化数据集


    combine_list = [Combine1, Combine2, Combine3]
    datalengths_list = [3000,2000,5000]

  

    dataSetList, mu_list0, sigma_list0 = loadData(combine_list, datalengths_list)

    #plotData(dataSetList,datalengths_list)   用来画图

    alpha_list = [0.33, 0.33, 0.34]

    data_list = [0.3,0.2,0.5]
    


    #开始EM算法，进行参数估计

    alpha_list, mu_list, sigma_list = EM_Train(dataSetList, mu_list0, sigma_list0, alpha_list)
    


    #打印参数预测结果
    print('the data parameters is :',data_list)
    print('----------------------------')
    print('the Parameters predict is:')
    print("alpha_list, mu_list, sigma_list____:",alpha_list, mu_list, sigma_list)


    #打印时间
    print('----------------------------')
    print('time span:', time.time() - start)






    


                   
