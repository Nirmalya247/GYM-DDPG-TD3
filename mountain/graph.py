import numpy as np
import matplotlib.pyplot as plt
file1 = open('./ddpg1.txt', 'r')
ddpg = file1.readlines()
file1 = open('./td3.txt', 'r')
td3 = file1.readlines()

def getDataDDPG(x):
    return float(x.split(",")[0].strip())

def getDataTD3(x):
    return float(x.split(",")[0].strip())

ddpg = [getDataDDPG(i) for i in ddpg]
td3 = [getDataTD3(i) for i in td3]


xs = np.array([i for i in range(len(ddpg))])
ys = np.array(ddpg)
plt.plot(xs, ys, label='DDPG')
plt.legend()
plt.draw()
xs = np.array([i for i in range(len(td3))])
ys = np.array(td3)
plt.plot(xs, ys, label='TD3')
plt.legend()
plt.draw()
plt.show()