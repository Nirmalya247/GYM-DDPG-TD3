import numpy as np
import matplotlib.pyplot as plt
file1 = open('./reward.txt', 'r')
td3 = file1.readlines()

def getDataTD3():
    return [x.split(",")[2].split(":")[1].strip() for x in td3 if x.find("Episode:") >= 0]

# ddpg = [getDataDDPG(i) for i in ddpg]
td3 = getDataTD3()
td32 = [ ]
for i in range(1, len(td3), 100):
    td32.append(td3[i])


xs = np.array([i for i in range(len(td32))])
ys = np.array(td32)
plt.plot(xs, ys, label='TD3')
plt.legend()
plt.draw()
plt.show()