from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import numpy as np


matrix = np.genfromtxt('data.txt', delimiter = ',')
m=0
for m in range(297):
    if matrix[m][13] != 0:
        matrix[m][13] = 1

ds = SupervisedDataSet(13,1)
net = buildNetwork(13,15,1)

i=0
for i in range(178):    
    ds.addSample((matrix[i][0], matrix[i][1],matrix[i][2],matrix[i][3],
                    matrix[i][4],matrix[i][5],matrix[i][6],matrix[i][7],
                    matrix[i][8],matrix[i][9],matrix[i][10],matrix[i][11],
                    matrix[i][12]),(matrix[i][13]))
                    

trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(50)

i=0
for i in range(237):
    print net.activate([matrix[i][0], matrix[i][1],matrix[i][2],matrix[i][3],
                    matrix[i][4],matrix[i][5],matrix[i][6],matrix[i][7],
                    matrix[i][8],matrix[i][9],matrix[i][10],matrix[i][11],
                    matrix[i][12]]),i