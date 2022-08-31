import numpy as np
import matplotlib.pyplot as plt
file1 = open('./ddpg1.txt', 'r')
ddpg = file1.readlines()

def getDataDDPG(x):
    return float(x.split(",")[0].strip())

ddpg = [getDataDDPG(i) for i in ddpg][0 : 133]


xs = np.array([i for i in range(len(ddpg))])
ys = np.array(ddpg)
plt.plot(xs, ys, label='DDPG')
plt.legend()
plt.draw()
plt.show()