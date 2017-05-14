import numpy as np
from sklearn.decomposition import TruncatedSVD
import marshal

d = marshal.load(open("../ip_mat", 'rb'))

X = []
l = []
for x in d:
    l.append(x)
    X.append(d[x])
X = np.array(X)

svd = TruncatedSVD(n_components = 16, random_state=42)
X = svd.fit_transform(X)

print(X.shape)

for i in range(len(l)):
    d[l[i]] = int(np.argmax(X[i]))
marshal.dump(d,open("../testcate","wb"))