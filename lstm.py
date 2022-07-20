import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/home/devin/share/manager/DDPG")
from ddpg import buffer
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import logging
import threading
import pickle
import dask.dataframe as dd
import visdom
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma
def init_line(name,windowName:list):
    try:
        vis = visdom.Visdom(env=name)
        for wn in windowName:
            if vis.win_exists(wn):
                print("win:{}存在！".format(wn))
                vis.close(wn)
            else:
                print("win{}不存在".format(wn))
        
        # vis.delete_env(name)
        return vis
    except Exception as e:
        print(str(e))
def draw_line(vis, win, step, lines, names):
    # draw multiple lines in one panel.
    """
    :param vis: the object of the visdom.Visdom
    :param step: the step of the line
    :param lines: the lines tuple, (line1, line2, ...)
    :param names: the names tuple, (name1, name2, ...)
    :return: None
    """
    if not len(lines) == len(names):
        raise ValueError('The length of the input is not the same')

    for line, name in zip(lines, names):
        vis.line(X=torch.Tensor([step]),
                 Y=torch.Tensor([line]),
                 win=win,
                 name='%s' % name,
                 update='append',
                 opts=dict(
                     title = win,
                     showlegend=True)
                 )

def shutdown():
    raise Exception()
def reshape(input):
    inputReshape = torch.empty((input.size(dim=1),input.size(dim=0),1))
    for index,_input in enumerate(input.split(1,dim = 1)):
        inputReshape[index] = _input
    inputReshape = inputReshape.double()
    return inputReshape
def reshape1(input):
    result = torch.empty((input.size(dim=1),input.size(dim=0)))
    for index, _input in enumerate(input.split(1,dim=0)):
        result[:,index] = _input[:,0]
    return result

class Sequence(nn.Module):
    def __init__(self,input_size,numofHiddenLayer,hidden_size,out_size):
        super(Sequence, self).__init__()
        # self.rnn = nn.LSTMCell(input_size,hidden_size)
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = numofHiddenLayer,
            batch_first = True,
        )
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, input, h_n, h_c):
        # print(1,input[:])
        # r_out, (h_n, h_c) = self.rnn(input, h_n, h_c)
        # NOTE: N = batch size
        #       L = sequence length
        #       D = 2 if bidirectional = True otherwise 1 
        #       H_in = input_size
        #       H_cell = hidden_size
        #       H_out = proj_size if proj_size > 0 otherwise hidden_size
        # NOTE: output: (N, L, H_out) hn: (num_layers, N, H_out) cn: (num_layers, N, H_cell) 
        output, (h_n,h_c) = self.rnn(input, (h_n, h_c)) 
        # print(output)
        out = self.linear(output[:,-1,:])
        return out,h_n,h_c
class lstm():
    def __init__(self,input_size,hidden_size,out_size,learningRateLstm,batchSize,ramSize):
        self.logger = logging.getLogger("specmanager.LSTM")
        self.lock = threading.Lock()
        self.numofHiddenLayer = 2
        self.sequence = Sequence(input_size,self.numofHiddenLayer,hidden_size,out_size).double()
        self.hidden_size = hidden_size
        self.h_n = torch.randn(self.numofHiddenLayer,1,hidden_size).double() # numofLayers,batch size,hiddenSize
        self.h_c = torch.randn(self.numofHiddenLayer,1,hidden_size).double()
        self.criterion = nn.MSELoss()
        self.lastOut = None
        self.batchSize = batchSize
        self.ramSize = ramSize
        self.ram = buffer.MemoryBuffer(self.ramSize)
        # use LBFGS as optimizer since we can load the whole data to train
        # optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
        self.optimizer = optim.Adam(self.sequence.parameters(), lr = learningRateLstm)

    def predict(self,state:list):
        """ 对输入的state进行预测，此处假设state是一个一维的list变量 """
        state = torch.tensor(state).unsqueeze(0).unsqueeze(0)
        state = Variable(state).double()
        # self.logger.info("输入LSTM的state的形状是{}的".format(state.shape))
        self.lock.acquire()
        self.out,self.h_n,self.h_c = self.sequence(state,self.h_n,self.h_c)
        self.lock.release()
        return self.out

    def optimize(self):
        self.optimizer.zero_grad()
        QueLength = self.ram.getNotNoneAllQue()
        if QueLength:
            QueLengthTensor = torch.tensor(QueLength).unsqueeze(0).unsqueeze(0).double()
            self.lock.acquire()
            out,self.h_n,self.h_c = self.sequence(QueLengthTensor,self.h_n,self.h_c)
            loss = criterion(out[:-1],QueLengthTensor[1:])
            loss.step()
            self.lock.release()
        else:
            # self.logger.debug("LSTM中的buffer太短了不优化")
            self.logger.debug("LSTM's memory too small and will do nothing!")

    def load_state_dict(self):
        self.lock.acquire()
        paramDict = self.sequence.state_dict()
        self.lock.release()
        return paramDict
    def load_state_dict_hex(self):
        paramDict = self.load_state_dict()
        return pickle.dumps(paramDict).hex()
    def set_state_dict(self,dict):
        self.lock.acquire()
        self.sequence.load_state_dict(dict)
        self.lock.release()
    def store_transition(self, trans):
        self.ram.add(trans)




if __name__ == '__main__':
    vis = init_line("lstm",["loss1","loss2","loss3"])
    # # set random seed to 0
    # np.random.seed(0)
    # torch.manual_seed(0)

    # load data and make training set
    ddf = dd.read_csv("trainingdata.csv")
    data = np.array(ddf.up)
    print("处理数据")
    # data = standardization(data)
    dataSize = len(data)





    hidden_size = 16
    seq = Sequence(1,2,hidden_size,1).double().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(seq.parameters(), lr = 0.0001)


    #begin to train
    batchSize = 100
    preSeqSize = 100

    steps = dataSize//batchSize
    print("总数据长度",dataSize)
    print("总共要运行{}个回合".format(steps))
    for i in range(steps):
        indexTemp = np.random.choice(dataSize-preSeqSize, batchSize)
        input = np.stack([data[idx:idx+preSeqSize].reshape(-1,1) for idx in indexTemp],axis=0)
        input = torch.tensor(input).cuda()

        target = np.stack([data[idx+preSeqSize] for idx in indexTemp],0)
        target = torch.tensor(target).unsqueeze(1).cuda()

        ## 删除0数据
        boolTensor = torch.BoolTensor(np.ones(batchSize))

        for subBatch in range(batchSize):
            if input[subBatch].isnan().any() or target[subBatch].isnan().any():
                boolTensor[subBatch] = False

        input_no_nan = input[boolTensor]
        target_no_nan = target[boolTensor]
        batch_no_nan = torch.count_nonzero(boolTensor)
        if batch_no_nan == 0:
            print("空的Batch")
        else:
        # print(input_no_nan.shape,target_no_nan.shape)
        # break

            optimizer.zero_grad()
            hc = torch.zeros(2,batch_no_nan,hidden_size).double().cuda()
            hn = torch.zeros(2,batch_no_nan,hidden_size).double().cuda()
            out,_,_ = seq(input_no_nan,hn,hc)
            # print(out)
            loss = criterion(out,target_no_nan)
            draw_line(vis,'loss1',i,(loss.item(),),["loss"])
            print("iter:{},loss:{}".format(i,loss.item()))
            loss.backward()
            optimizer.step()
