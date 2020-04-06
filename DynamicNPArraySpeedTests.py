
from skeLCS.DynamicNPArray import TupleArray,ListArray
import numpy as np
import time
import random

class Arb(): #Test Object
    def __init__(self,num):
        self.num = 5

    def __eq__(self, other):
        if not isinstance(other,Arb):
            return False

        return other.num == self.num

#Initialization
print()
t = time.time()
d = TupleArray(k=1)
print("Init empty ta: "+str(time.time()-t))

t = time.time()
d = TupleArray(np.array([1,3,6,5,4,7,8,9,10]))
print("Init full ta: "+str(time.time()-t))

t = time.time()
d = ListArray(k=1)
print("Init empty la: "+str(time.time()-t))

t = time.time()
d = ListArray(np.array([1,3,6,5,4,7,8,9,10]))
print("Init full la: "+str(time.time()-t))

t = time.time()
d = np.array([])
print("Init empty npa: "+str(time.time()-t))

t = time.time()
d = np.array([1,3,6,5,4,7,8,9,10])
print("Init full npa: "+str(time.time()-t))

t = time.time()
d = []
print("Init empty list: "+str(time.time()-t))

t = time.time()
d = [1,3,6,5,4,7,8,9,10]
print("Init full list: "+str(time.time()-t))


#Append
print()
d = TupleArray(k=1)
t = time.time()
for i in range(100000):
    d.append(random.randint(1,1000))
print("append ta: "+str(time.time()-t))

d = ListArray(k=1)
t = time.time()
for i in range(100000):
    d.append(random.randint(1,1000))
print("append la: "+str(time.time()-t))

d = np.array([])
t = time.time()
for i in range(100000):
    d = np.append(d,random.randint(1,1000))
print("append npa: "+str(time.time()-t))

d = []
t = time.time()
for i in range(100000):
    d.append(random.randint(1,1000))
print("append list: "+str(time.time()-t))

#AppendObj
print()
d = TupleArray(k=1,dtype=Arb)
t = time.time()
for i in range(10000):
    d.append(Arb(random.randint(1,1000)))
print("appendObj ta: "+str(time.time()-t))

d = ListArray(k=1)
t = time.time()
for i in range(10000):
    d.append(Arb(random.randint(1,1000)))
print("appendObj la: "+str(time.time()-t))

d = np.array([])
t = time.time()
for i in range(10000):
    d = np.append(d,Arb(random.randint(1,1000)))
print("appendObj npa: "+str(time.time()-t))

d = []
t = time.time()
for i in range(10000):
    d.append(Arb(random.randint(1,1000)))
print("appendObj list: "+str(time.time()-t))


#Append2D
print()
d = TupleArray(k=2)
t = time.time()
for i in range(100000):
    d.append(np.array([random.randint(1,1000),random.randint(1,1000)]))
print("append2D ta: "+str(time.time()-t))

d = ListArray(k=2)
t = time.time()
for i in range(100000):
    d.append(np.array([random.randint(1,1000),random.randint(1,1000)]))
print("append2D la: "+str(time.time()-t))

d = np.array([[0,0]])
t = time.time()
for i in range(100000):
    d = np.concatenate((d,np.array([[random.randint(1,1000),random.randint(1,1000)]])),axis=0)
print("append2D npa: "+str(time.time()-t))

d = []
t = time.time()
for i in range(100000):
    d.append([random.randint(1,1000),random.randint(1,1000)])
print("append2D list: "+str(time.time()-t))

#Index
print()
d = TupleArray(k=1)
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    d.getI(i)
print("index ta: "+str(time.time()-t))

d = TupleArray(k=1)
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    d.a[i,0]
print("index a ta: "+str(time.time()-t))

d = TupleArray(k=1)
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    d.getArray()[i]
print("index getArray ta: "+str(time.time()-t))

d = ListArray(k=1)
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    d.getI(i)
print("index la: "+str(time.time()-t))

d = np.array([])
for i in range(100000):
    d = np.append(d,random.randint(1,1000))
t = time.time()
for i in range(100000):
    d[i]
print("index npa: "+str(time.time()-t))

d = []
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    d[i]
print("index list: "+str(time.time()-t))

#RemoveLast
print()
d = TupleArray(k=1)
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    d.removeLast()
print("removeLast ta: "+str(time.time()-t))

d = ListArray(k=1)
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    d.removeLast()
print("removeLast la: "+str(time.time()-t))

d = np.array([])
for i in range(100000):
    d = np.append(d,random.randint(1,1000))
t = time.time()
for i in range(100000):
    d = np.delete(d,-1)
print("removeLast npa: "+str(time.time()-t))

d = []
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    d.pop()
print("removeLast list: "+str(time.time()-t))

#RemoveLast2D
print()
d = TupleArray(k=2)
for i in range(100000):
    d.append(np.array([random.randint(1,1000),random.randint(1,1000)]))
t = time.time()
for i in range(100000):
    d.removeLast()
print("removeLast2D ta: "+str(time.time()-t))

d = ListArray(k=2)
for i in range(100000):
    d.append(np.array([random.randint(1,1000),random.randint(1,1000)]))
t = time.time()
for i in range(100000):
    d.removeLast()
print("removeLast2D la: "+str(time.time()-t))

d = np.array([[0,0]])
for i in range(100000):
    d = np.concatenate((d,np.array([[random.randint(1,1000),random.randint(1,1000)]])),axis=0)
t = time.time()
for i in range(100000):
    d = np.delete(d,-1)
print("removeLast2D npa: "+str(time.time()-t))

d = []
for i in range(100000):
    d.append([random.randint(1,1000),random.randint(1,1000)])
t = time.time()
for i in range(100000):
    d.pop()
print("removeLast2D list: "+str(time.time()-t))

#RemoveIndex
print()
d = TupleArray(k=1)
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000-1):
    d.removeAtIndex(1)
print("removeIndex ta: "+str(time.time()-t))

d = ListArray(k=1)
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000-1):
    d.removeAtIndex(1)
print("removeIndex la: "+str(time.time()-t))

d = np.array([])
for i in range(100000):
    d = np.append(d,random.randint(1,1000))
t = time.time()
for i in range(100000-1):
    d = np.delete(d,1)
print("removeIndex npa: "+str(time.time()-t))

d = []
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000-1):
    del d[1]
print("removeIndex list: "+str(time.time()-t))

#np 1D operations
print()
d = TupleArray(k=1)
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    d = TupleArray(d.getArray()+3)
print("1D ops ta: "+str(time.time()-t))

d = ListArray(k=1)
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    d = ListArray(d.getArray()+3)
print("1D ops la: "+str(time.time()-t))

d = np.array([])
for i in range(100000):
    d = np.append(d,random.randint(1,1000))
t = time.time()
for i in range(100000):
    d = d+3
print("1D ops npa: "+str(time.time()-t))

d = []
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    for n in d:
        n+=3
print("1D ops list: "+str(time.time()-t))

d = []
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    n = np.array(d)
    n+=3
    d = n.tolist()
print("1D ops list conversion: "+str(time.time()-t))

#np 2D operations
print()
d = TupleArray(k=2)
for i in range(100000):
    d.append(np.array([random.randint(1,1000),random.randint(1,1000)]))
t = time.time()
for i in range(100000):
    d = TupleArray(d.getArray()+3)
print("2D ops ta: "+str(time.time()-t))

d = ListArray(k=2,minSize=1000)
for i in range(100000):
    d.append(np.array([random.randint(1,1000),random.randint(1,1000)]))
t = time.time()
for i in range(100000):
    d = ListArray(d.getArray()+3)
print("2D ops la: "+str(time.time()-t))

d = np.array([[0,0]])
for i in range(100000):
    d = np.concatenate((d,np.array([[random.randint(1,1000),random.randint(1,1000)]])),axis=0)
t = time.time()
for i in range(100000):
    d = d+3
print("2D ops npa: "+str(time.time()-t))

d = []
for i in range(100000):
    d.append([random.randint(1,1000),random.randint(1,1000)])
t = time.time()
for i in range(100000):
    for n in d:
        for j in n:
            j+=3
print("2D ops list: "+str(time.time()-t))

d = []
for i in range(100000):
    d.append(random.randint(1,1000))
t = time.time()
for i in range(100000):
    n = np.array(d)
    n+=3
    d = n.tolist()
print("2D ops list conversion: "+str(time.time()-t))