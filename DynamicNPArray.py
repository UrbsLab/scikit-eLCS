
import numpy as np

class ArrayFactory():
    def createArray(a=np.array([]),minSize = 8, k = np.nan, dtype =  np.float64):
        return TupleArray(a,minSize=minSize,k=k,dtype=dtype)

class GenericArray():
    def __init__(self):
        pass

class TupleArray(GenericArray):
    '''
    A TupleArray is a data structure that can be thought of as a 2D array, where axis = 0 is mutable in size, and
    axis = 1 is immutable in size.

    This object was created using numpy ndarrays, and is useful when a user requires the functionality and invariants
    of an ndarray, but the fast size mutability of a python list (in one direction).

    For example, a user may want to store an expanding list of 5 specific numeric attributes. In the context of eLCS,
    TupleArrays can be widely used in the storage of rules.

    Aspirationally, this structure can be extended to support fast mutability in multiple axis. If this is made possible,
    the structure can conceivably be used as a universal replacement for ndarrays. However, this is currently not
    implemented (see the commented out attempt below). Issues w/ the necessary 0s concatenation prevent n-dimensional
    mutability at the moment.
    '''
    def __init__(self,a=np.array([]),minSize = 8,k = np.nan, dtype = np.float64):
        '''
        :param a: This can either be a shape (n,) ndarray, or a shape (n,k) ndarray. The (n,) will be transformed to a
                  (n,1) array for operations
        :param minSize: minimum size of n in terms of memory allocated
        :param k: explicit setting of dimension (only used and mandatory if array is initialized empty)
        :param dtype: explicit setting of array dtype (only used if array is initialized empty)
        '''

        self.minSize = minSize
        if not(isinstance(a,np.ndarray) and len(a.shape) <= 2) or (len(a.shape) == 2 and (a.shape[1] == 0 or a.shape[0] == 0)):
            raise Exception("Invalid input array: must be ndarray and have dimension <= 2")

        if len(a.shape) == 1 and not(np.array_equal(a,np.array([]))):
            self.transposed = True
            a = np.transpose([a]) #Enforce uniform shape of (n,k) for all inputs
        elif np.array_equal(a,np.array([])) and k == 1:
            self.transposed = True
        else:
            self.transposed = False

        if minSize < 1:
            raise Exception("minSize must be >= 1")
        if np.array_equal(a,np.array([])):
            self.dtype = dtype
            if k < 1 or np.isnan(k):
                raise Exception("Must specify valid dimension if you init w/ empty array")
            else:
                self.k = k

            if np.issubdtype(dtype, np.number):
                self.emptyPrefix = 0
            else:
                self.emptyPrefix = None
        else:
            self.dtype = a.dtype
            self.k = a.shape[1]
            if np.issubdtype(a.dtype, np.number):
                self.emptyPrefix = 0
            else:
                self.emptyPrefix = None


        self.lastIndex = a.shape[0]-1
        upperBound = self.minSize
        while (self.lastIndex + 1 > upperBound):
            upperBound *= 2

        empty = np.full((upperBound-self.lastIndex-1,self.k),self.emptyPrefix,dtype=self.dtype)
        if np.array_equal(a,np.array([])):
            self.a = empty
        else:
            self.a = np.concatenate((a,empty),axis=0)

    def append(self,value):
        if self.lastIndex == self.a.shape[0] - 1:
            empty = np.full(self.a.shape,self.emptyPrefix,dtype=self.dtype)
            self.a = np.concatenate((self.a,empty,),axis=0)
        self.lastIndex+=1
        self.a[self.lastIndex] = value

    def removeLast(self):
        if (self.lastIndex == -1):
            raise Exception("Array size is already zero. Cannot remove.")
        self.lastIndex -= 1
        if self.lastIndex + 1 <= self.a.shape[0]/2 and self.a.shape[0] > self.minSize:
            self.a = self.a[:int(-self.a.shape[0]/2),:] #Half array if removed half

    def removeAtIndex(self,index):
        if (index < 0 or index > self.lastIndex):
            raise Exception('Index out of bounds')

        self.a = np.concatenate((self.a[0:index], self.a[index + 1:self.a.shape[0]], [np.array([self.emptyPrefix]*self.k)]), axis=0)

        self.lastIndex -= 1
        if self.lastIndex + 1 <= self.a.shape[0] / 2 and self.a.shape[0] > self.minSize:
            self.a = self.a[:int(-self.a.shape[0] / 2), :]  # Half array if removed half

    def removeFirstElementWithValue(self,value): #If doesn't exist, it'll just crash.
        if self.k != 1:
            raise Exception("This method is only useable for 1D arrays")
        index = np.where(self.a == value)[0][0]
        self.removeAtIndex(index)

    def getArray(self): #If at init, the array was transposed, returns a shape (k,) array
        if self.lastIndex == -1:
            return np.array([])
        if (self.transposed):
            return np.transpose(self.a[:self.lastIndex+1])[0]
        else:
            return self.a[:self.lastIndex+1]

    def size(self):
        return (self.lastIndex+1)*self.k

    def shape(self):
        if self.transposed:
            return (self.lastIndex+1,)
        else:
            return (self.lastIndex+1,self.k)

    def getI(self,x,y=0):
        return self.a[x,y]

    def setI(self,x,y=0,value=0):
        self.a[x,y] = value

    def setRowI(self,x,value=0): #value must be an np array
        self.a[x] = value

    def getRowI(self,x):
        return self.a[x]

class ListArray(GenericArray):

    def __init__(self,a=np.array([]),minSize = 8, k = np.nan, dtype = np.float64):
        '''
        :param a: This can either be a shape (n,) ndarray, or a shape (n,k) ndarray. The (n,) will be transformed to a (n,1) array for operations
        :param minSize: unnecessary, but existent to maintain interface
        :param k: explicit setting of dimension (only used and mandatory if array is initialized empty)
        :param dtype: unnecessary, but existent to maintain interface
        '''

        if not(isinstance(a,np.ndarray) and len(a.shape) <= 2) or (len(a.shape) == 2 and (a.shape[1] == 0 or a.shape[0] == 0)):
            raise Exception("Invalid input array: must be ndarray and have dimension <= 2")

        if len(a.shape) == 1 and not(np.array_equal(a,np.array([]))):
            self.transposed = True
            a = np.transpose([a]) #Enforce uniform shape of (n,k) for all inputs
        elif np.array_equal(a,np.array([])) and k == 1:
            self.transposed = True
        else:
            self.transposed = False

        if np.array_equal(a,np.array([])):
            if k < 1 or np.isnan(k):
                raise Exception("Must specify valid dimension if you init w/ empty array")
            else:
                self.k = k
        else:
            self.k = a.shape[1]

        self.a = a.tolist()

    def append(self,value): #currently only supports 1D and 2D arrays along axis 0

        if self.k==1:
            self.a.append([value])
        else:
            if not (isinstance(value, np.ndarray)):
                raise Exception("Must be an nd array")
            self.a.append(value.tolist())

    def removeLast(self):
        self.a.pop()

    def removeAtIndex(self,index):
        del self.a[index]

    def removeFirstElementWithValue(self,value):
        if self.k != 1:
            raise Exception("This method is only useable for 1D arrays")
        self.a.remove([value])

    def getArray(self):
        if (self.transposed and self.a != []):
            return np.transpose(np.array(self.a))[0]
        else:
            return np.array(self.a)

    def shape(self):
        return tuple(len(self.a),self.k)

    def size(self):
        return len(self.a)*self.k

    def getI(self, x, y=0):
        return self.a[x][y]

    def setI(self, x, y=0, value=0):
        self.a[x][y] = value

    def setRowI(self,x,value=0): #value must be an np array
        self.a[x] = value.tolist()

    def getRowI(self,x):
        return np.array(self.a[x])

"""
class DynamicNPArray():

    def __init__(self,a=np.array([])): #a must be an np array (can be nD)

        self.dimensionCount = len(a.shape)

        #Determine last indexes
        self.lastIndices = []
        for dimension in range(self.dimensionCount):
            self.lastIndices.append(a.shape[dimension]-1)

        #Determine inital upper bounds
        self.upperBounds = []
        for dimension in range(self.dimensionCount):
            upperBound = 16
            while (self.lastIndices[dimension]+1 > upperBound):
                upperBound *= 2
            self.upperBounds.append(upperBound)

        #Set up bounding array
        self.a = a
        for dimension in range(self.dimensionCount):
            emptyA = []
            for dimension2 in range(self.dimensionCount):
                if dimension2 < dimension:
                    emptyA.append(self.upperBounds[dimension2])
                elif dimension2 == dimension:
                    emptyA.append(self.upperBounds[dimension2]-a[dimension2])
                else:
                    emptyA.append(a[dimension2])
            empty = np.zeros(tuple(emptyA))
            self.a = np.concatenate((self.a,empty),axis=dimension)


    '''
    IMPORTANT NOTE: While the initializer supports the input of any n dimensional arrays, the following functions only
    support 1D arrays, and 2D arrays that operate on axis = 0:
    -append
    -removeLast
    -removeAtIndex

    The following function only support 1D arrays:
    -removeFirstElementWithValue

    Future versions may extend functionality to all nD arrays
    '''
    '''
    def append(self,add,axis=0): #axis sizes must match all except current axis. Can only add one at a time. add can be either ndarray or value
        if isinstance(add,np.ndarray):
            if add.shape[axis] != 1 or axis >= self.dimensionCount:
                raise Exception("Appending axis size must be 1")
            isShapeMatch = True


            if self.lastIndices[axis] == self.a.shape[axis]-1:
                e = np.zeros(self.a.shape)
                self.a = np.concatenate((self.a,e),axis=axis)
            self.lastIndices[axis]+=1

            if self.dimensionCount == 1:
                self.a[self.lastIndices[axis]] = add
            elif self.dimensionCount == 2 and axis == 0:
                self.a[self.lastIndices[axis]] = add
            elif self.dimensionCount == 2 and axis == 1:
                self.a[:,self.lastIndices[axis]] = add
            else: #arrays above 2D operate suboptimally in terms of runtime at the moment
                s = ""
                for dimension in range(self.dimensionCount):
                    if dimension != axis:
                        s+=":"
                    else:
                        s+=str(self.lastIndices[axis])
                    if dimension != self.dimensionCount-1:
                        s+=','
                slice_ = eval(f'np.s_[{s}]')
                self.a[slice_] = add
        else:
            if self.dimensionCount != 1:
                raise Exception("Cannot append singular value to multi dimensional array")



        if self.lastIndex == self.a.shape[0]-1: #Double size if out of bounds
            e = np.zeros((self.a.shape[0], self.state))
            self.a = np.concatenate((self.a,e),axis=0)
        self.lastIndex += 1
        self.a[self.lastIndex] = add
    '''
    def append(self,add):
        '''add can either be a single value, or for 2D arrays, a 1D array of shape (n,) where n is the shape of axis 1
        (eventually converts to (1,n) shape for concatenation via adding of [])
        '''
        if isinstance(add, np.ndarray):#For 2D arrays
            if len(add.shape) != 1 or add.shape[0] != self.lastIndices[1]:
                raise Exception("Invalid ndarray to append: size mismatch")
            e = np.zeros((self.upperBounds))
        else:



    def removeLast(self):
        self.lastIndex -= 1
        if self.lastIndex + 1 <= self.a.shape[0]/2 and self.a.shape[0] > 16:
            self.a = self.a[:-self.a.shape[0]/2,:] #Half array if removed half

    def removeAtIndex(self,index):
        l = list(self.a)
        del l[index]
        self.a = np.array(l)

        self.lastIndex -= 1
        if self.lastIndex + 1 <= self.a.shape[0] / 2 and self.a.shape[0] > 16:
            self.a = self.a[:-self.a.shape[0] / 2, :]  # Half array if removed half

    def removeFirstElementWithValue(self,value):
        index = np.where(self.a == value)[0][0]
        self.removeAtIndex(index)

    def getArray(self):
        return self.a[:self.lastIndex+1]

    def size(self):
        return np.prod(np.array(self.lastIndices)+1)

    def shape(self):
        return tuple((np.array(self.lastIndices)+1).tolist())
"""
