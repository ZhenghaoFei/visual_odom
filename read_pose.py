import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pose = np.loadtxt('/Users/holly/Downloads/KITTI/poses/00.txt')
print(pose.shape)

pose = pose[:1000]
x = pose[:, 3]
y = pose[:, 7]
z = pose[:, 11]

plt.plot(x, z)
plt.show()