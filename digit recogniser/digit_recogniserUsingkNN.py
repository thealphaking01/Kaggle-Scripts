import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

train = np.genfromtxt("train.csv",skip_header=True,delimiter=",")
test = np.genfromtxt("test.csv",skip_header=True,delimiter=",")

label = train[:,0]
train = train [:,1:]

print (len(label))
digit = []

def euclid(a,b):
    a = np.array(a, dtype="float")
    b = np.array(b, dtype="float")
    return np.sum(np.power(np.subtract(a,b),2))

ans = []

for i in test:
    count = {}
    dist = []
    l=0
    for j in train:
        dist.append((label[l],euclid(i,j)))
        l+=1
    dist = sorted(dist, key=lambda x: x[1])[:20]
    for i,j in dist:
        count[i]=count.get(i,0)+1
    count = sorted(count, key=lambda k: count[k])
    ans.append(count[0])

l= len(test)
imageId = [ i for i in range(1,l+1)]
submission = pd.DataFrame({
        "ImageId": imageId,
        "Label": ans
    })

submission.to_csv("kaggle.csv", index=False)
