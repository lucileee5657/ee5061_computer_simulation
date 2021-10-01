import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

result = np.array([])
with open("truncated_normal.txt", "r") as f:
    l = f.readline()
    while l:
        l = l.strip()
        l = float(l)
        result = np.append(result, l)
        l = f.readline()
#print(result)

plt.hist(result, color = 'blue', edgecolor = 'black',
         bins = int(20/0.1))
plt.title('Histogram of truncated normal')
#plt.xlabel('how many')
#plt.ylabel('random numbers')
plt.show()