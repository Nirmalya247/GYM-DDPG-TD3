import numpy as np
import matplotlib.pyplot as plt
file1 = open('./ddpg1_old.txt', 'r')
ddpg = file1.readlines()

def getDataDDPG(x):
    return float(x.split(",")[0].strip()) / float(x.split(",")[2].strip())

ddpg = [getDataDDPG(i) for i in ddpg]
ddpg2 = [ ]
for i in range(1, len(ddpg), 100):
    ddpg2.append(ddpg[i])


xs = np.array([i for i in range(len(ddpg2))])
ys = np.array(ddpg2)
plt.plot(xs, ys, label='DDPG')
plt.legend()
plt.draw()
plt.show()