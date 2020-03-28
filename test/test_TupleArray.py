import unittest
import numpy as np
from skeLCS.DynamicNPArray import TupleArray

class Arb(): #Test Object
    def __init__(self,num):
        self.num = 5

    def __eq__(self, other):
        if not isinstance(other,Arb):
            return False

        return other.num == self.num

class TestTupleArray(unittest.TestCase):

    def testEmpty1DInit(self):
        t = TupleArray(k=1)
        self.assertTrue(np.array_equal(np.array([[0],[0],[0],[0],[0],[0],[0],[0]]),t.a))
        self.assertTrue(np.array_equal(np.array([]),t.getArray()))
        self.assertEqual(-1,t.lastIndex)
        self.assertEqual(0,t.size())

    def testSmall1DInit(self):
        t = TupleArray(np.array([2,3,6,5]))
        c = np.array([[2],[3],[6],[5],[0],[0],[0],[0]])
        self.assertTrue(np.array_equal(c,t.a))
        self.assertTrue(np.array_equal(np.array([2,3,6,5]),t.getArray()))
        self.assertEqual(3, t.lastIndex)
        self.assertEqual(4, t.size())

    def testSmall1DInit2(self):
        t = TupleArray(np.array([2,3,6,5]),minSize=4)
        c = np.array([[2],[3],[6],[5]])
        self.assertTrue(np.array_equal(c,t.a))
        self.assertTrue(np.array_equal(np.array([2,3,6,5]),t.getArray()))
        self.assertEqual(3, t.lastIndex)
        self.assertEqual(4, t.size())

    def testLarge1DInit(self):
        t = TupleArray(np.array([2, 3, 6, 5,4,5,1,2,5]))
        c = np.array([[2], [3], [6], [5], [4], [5], [1], [2], [5],[0],[0],[0],[0],[0],[0],[0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([2, 3, 6, 5,4,5,1,2,5]), t.getArray()))
        self.assertEqual(8, t.lastIndex)
        self.assertEqual(9, t.size())

    def testLarge1DInit2(self):
        t = TupleArray(np.array([2, 3, 6, 5, 4]),minSize=4)
        c = np.array([[2], [3], [6], [5], [4], [0], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([2, 3, 6, 5, 4]), t.getArray()))
        self.assertEqual(4, t.lastIndex)

    def testSmall2DInit(self):
        t = TupleArray(np.array([[2, 3],[1, 8],[9, 1],[6, 7],[3, 1]]))
        c = np.array([[2, 3],[1, 8],[9, 1],[6, 7],[3, 1],[0,0],[0,0],[0,0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[2, 3],[1, 8],[9, 1],[6, 7],[3, 1]]),t.getArray()))
        self.assertEqual(4, t.lastIndex)

    def testLarge2DInit(self):
        t = TupleArray(np.array([[2, 3],[1, 8],[9, 1],[6, 7],[3, 1]]),minSize=2)
        c = np.array([[2, 3],[1, 8],[9, 1],[6, 7],[3, 1],[0,0],[0,0],[0,0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[2, 3],[1, 8],[9, 1],[6, 7],[3, 1]]),t.getArray()))
        self.assertEqual(4, t.lastIndex)

    def testStrange1DInit(self):
        t = TupleArray(np.array([[2],[3],[4]]))
        c = np.array([[2],[3],[4],[0],[0],[0],[0],[0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[2],[3],[4]]), t.getArray()))
        self.assertEqual(2, t.lastIndex)

    def testEmptyObjInit(self):
        t = TupleArray(k=1,dtype=Arb)
        self.assertTrue(np.array_equal(np.array([[None], [None], [None], [None], [None], [None], [None], [None]]), t.a))
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(-1, t.lastIndex)
        self.assertEqual(0, t.size())

    def testSmall1DObjInit(self):
        t = TupleArray(np.array([Arb(2),Arb(3),Arb(6),Arb(5)]))
        c = np.array([[Arb(2)],[Arb(3)],[Arb(6)],[Arb(5)],[None],[None],[None],[None]])
        self.assertTrue(np.array_equal(c,t.a))
        self.assertTrue(np.array_equal(np.array([Arb(2),Arb(3),Arb(6),Arb(5)]),t.getArray()))
        self.assertEqual(3, t.lastIndex)
        self.assertEqual(4, t.size())

    def testLarge1DObjInit(self):
        t = TupleArray(np.array([Arb(2),Arb(3),Arb(6),Arb(5),Arb(6)]),minSize=4)
        c = np.array([[Arb(2)],[Arb(3)],[Arb(6)],[Arb(5)],[Arb(6)],[None],[None],[None]])
        self.assertTrue(np.array_equal(c,t.a))
        self.assertTrue(np.array_equal(np.array([Arb(2),Arb(3),Arb(6),Arb(5),Arb(6)]),t.getArray()))
        self.assertEqual(4, t.lastIndex)
        self.assertEqual(5, t.size())

    def testEmptyObj2DInit(self):
        t = TupleArray(k=2,dtype=Arb,minSize=4)
        self.assertTrue(np.array_equal(np.array([[None,None], [None,None], [None,None], [None,None]]), t.a))
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(-1, t.lastIndex)
        self.assertEqual(0, t.size())

    def testSmall2DObjInit(self):
        t = TupleArray(np.array([[Arb(2),Arb(1)],[Arb(3),Arb(0)],[Arb(6),Arb(7)],[Arb(5),Arb(70)]]))
        c = np.array([[Arb(2),Arb(1)],[Arb(3),Arb(0)],[Arb(6),Arb(7)],[Arb(5),Arb(70)],[None,None], [None,None], [None,None], [None,None]])
        self.assertTrue(np.array_equal(c,t.a))
        self.assertTrue(np.array_equal(np.array([[Arb(2),Arb(1)],[Arb(3),Arb(0)],[Arb(6),Arb(7)],[Arb(5),Arb(70)]]),t.getArray()))
        self.assertEqual(3, t.lastIndex)
        self.assertEqual(8, t.size())

    def testLarge2DObjInit(self):
        t = TupleArray(np.array([[Arb(2), Arb(1)], [Arb(3), Arb(0)], [Arb(6), Arb(7)], [Arb(5), Arb(70)],[Arb(30),Arb(20)]]),minSize=4)
        c = np.array([[Arb(2), Arb(1)], [Arb(3), Arb(0)], [Arb(6), Arb(7)], [Arb(5), Arb(70)],[Arb(30),Arb(20)], [None, None],[None, None], [None, None]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[Arb(2), Arb(1)], [Arb(3), Arb(0)], [Arb(6), Arb(7)], [Arb(5), Arb(70)],[Arb(30),Arb(20)]]),t.getArray()))
        self.assertEqual(4, t.lastIndex)
        self.assertEqual(10, t.size())

    def testInvalidInput(self):
        with self.assertRaises(Exception) as context:
            t = TupleArray(3)

        self.assertTrue("Invalid input array: must be ndarray and have dimension <= 2" in str(context.exception))

    def testInvalidInput2(self):
        with self.assertRaises(Exception) as context:
            t = TupleArray(np.array([[[2,3],[3,4]],[[4,3],[2,0]]]))

        self.assertTrue("Invalid input array: must be ndarray and have dimension <= 2" in str(context.exception))

    def testInvalidInput3(self):
        with self.assertRaises(Exception) as context:
            t = TupleArray(np.array([]),minSize=0)

        self.assertTrue("minSize must be >= 1" in str(context.exception))

    def testInvalidInput4(self):
        with self.assertRaises(Exception) as context:
            t = TupleArray(k=-1)

        self.assertTrue("Must specify valid dimension if you init w/ empty array" in str(context.exception))

    def testInvalidInput5(self):
        with self.assertRaises(Exception) as context:
            t = TupleArray()

        self.assertTrue("Must specify valid dimension if you init w/ empty array" in str(context.exception))

    def testInvalidInput6(self):
        with self.assertRaises(Exception) as context:
            t = TupleArray(np.array([[]]))

        self.assertTrue("Invalid input array: must be ndarray and have dimension <= 2" in str(context.exception))

    def testInvalidInput7(self):
        with self.assertRaises(Exception) as context:
            t = TupleArray(np.array([[],[]]))

        self.assertTrue("Invalid input array: must be ndarray and have dimension <= 2" in str(context.exception))

    def testInvalidInput8(self):
        with self.assertRaises(Exception) as context:
            n = np.array([[2,3]])
            n = np.delete(n,0,axis=0)
            t = TupleArray(n)

        self.assertTrue("Invalid input array: must be ndarray and have dimension <= 2" in str(context.exception))

    ####################################################################################################################
    def test1DAppendAndRemoveLast(self):
        t = TupleArray(minSize=4,k=1)

        t.append(6)
        c = np.array([[6],[0],[0],[0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([6]),t.getArray()))
        self.assertEqual(0, t.lastIndex)
        self.assertEqual(1, t.size())

        t.append([8])
        c = np.array([[6], [8], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([6,8]), t.getArray()))
        self.assertEqual(1, t.lastIndex)
        self.assertEqual(2, t.size())

        t.append([9])
        c = np.array([[6], [8], [9], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([6,8,9]), t.getArray()))
        self.assertEqual(2, t.lastIndex)
        self.assertEqual(3, t.size())

        t.append([8])
        c = np.array([[6], [8], [9], [8]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([6,8,9,8]), t.getArray()))
        self.assertEqual(3, t.lastIndex)
        self.assertEqual(4, t.size())

        t.append([8])
        c = np.array([[6], [8], [9], [8], [8], [0], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([6, 8, 9, 8,8]), t.getArray()))
        self.assertEqual(4, t.lastIndex)
        self.assertEqual(5, t.size())

        t.removeLast()
        c = np.array([[6], [8], [9], [8]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([6, 8, 9, 8]), t.getArray()))
        self.assertEqual(3, t.lastIndex)
        self.assertEqual(4, t.size())

        t.removeLast()
        c = np.array([[6], [8], [9], [8]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([6, 8, 9]), t.getArray()))
        self.assertEqual(2, t.lastIndex)
        self.assertEqual(3, t.size())

        t.removeLast()
        c = np.array([[6], [8], [9], [8]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([6, 8]), t.getArray()))
        self.assertEqual(1, t.lastIndex)
        self.assertEqual(2, t.size())

        t.removeLast()
        c = np.array([[6], [8], [9], [8]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([6]), t.getArray()))
        self.assertEqual(0, t.lastIndex)
        self.assertEqual(1, t.size())

        t.removeLast()
        c = np.array([[6], [8], [9], [8]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(-1, t.lastIndex)
        self.assertEqual(0, t.size())

        with self.assertRaises(Exception) as context:
            t.removeLast()
        self.assertTrue("Array size is already zero. Cannot remove." in str(context.exception))

        t.append(20)
        c = np.array([[20], [8], [9], [8]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([20]), t.getArray()))
        self.assertEqual(0, t.lastIndex)
        self.assertEqual(1, t.size())

        t.append(30)
        c = np.array([[20], [30], [9], [8]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([20,30]), t.getArray()))
        self.assertEqual(1, t.lastIndex)
        self.assertEqual(2, t.size())

    def test2DAppendAndRemoveLast(self):
        t = TupleArray(minSize=4,k=2)
        t.append([6,1])
        c = np.array([[6,1],[0,0],[0,0],[0,0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[6,1]]),t.getArray()))
        self.assertEqual(0, t.lastIndex)
        self.assertEqual(2, t.size())

        t.append([8,2])
        c = np.array([[6,1], [8,2], [0,0], [0,0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[6,1], [8,2]]), t.getArray()))
        self.assertEqual(1, t.lastIndex)
        self.assertEqual(4, t.size())

        t.append([9,5])
        c = np.array([[6,1], [8,2], [9,5], [0,0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[6,1], [8,2], [9,5]]), t.getArray()))
        self.assertEqual(2, t.lastIndex)
        self.assertEqual(6, t.size())

        t.append([0,1])
        c = np.array([[6,1], [8,2], [9,5], [0,1]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[6,1], [8,2], [9,5], [0,1]]), t.getArray()))
        self.assertEqual(3, t.lastIndex)
        self.assertEqual(8, t.size())

        t.append([7,7])
        c = np.array([[6,1], [8,2], [9,5], [0,1], [7,7], [0,0], [0,0], [0,0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[6,1], [8,2], [9,5], [0,1], [7,7]]), t.getArray()))
        self.assertEqual(4, t.lastIndex)
        self.assertEqual(10, t.size())

        t.removeLast()
        c = np.array([[6, 1], [8, 2], [9, 5], [0, 1]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[6, 1], [8, 2], [9, 5], [0, 1]]), t.getArray()))
        self.assertEqual(3, t.lastIndex)
        self.assertEqual(8, t.size())

        t.removeLast()
        c = np.array([[6, 1], [8, 2], [9, 5], [0, 1]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[6, 1], [8, 2], [9, 5]]), t.getArray()))
        self.assertEqual(2, t.lastIndex)
        self.assertEqual(6, t.size())

        t.removeLast()
        c = np.array([[6, 1], [8, 2], [9, 5], [0, 1]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[6, 1], [8, 2]]), t.getArray()))
        self.assertEqual(1, t.lastIndex)
        self.assertEqual(4, t.size())

        t.removeLast()
        c = np.array([[6, 1], [8, 2], [9, 5], [0, 1]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[6, 1]]), t.getArray()))
        self.assertEqual(0, t.lastIndex)
        self.assertEqual(2, t.size())

        t.removeLast()
        c = np.array([[6, 1], [8, 2], [9, 5], [0, 1]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(-1, t.lastIndex)
        self.assertEqual(0, t.size())

        with self.assertRaises(Exception) as context:
            t.removeLast()
        self.assertTrue("Array size is already zero. Cannot remove." in str(context.exception))

        t.append([20,11])
        c = np.array([[20, 11], [8, 2], [9, 5], [0, 1]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[20, 11]]), t.getArray()))
        self.assertEqual(0, t.lastIndex)
        self.assertEqual(2, t.size())

        t.append([30,20])
        c = np.array([[20, 11], [30, 20], [9, 5], [0, 1]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[20, 11],[30,20]]), t.getArray()))
        self.assertEqual(1, t.lastIndex)
        self.assertEqual(4, t.size())

    def test1DRemoveAtIndex(self):
        t = TupleArray(np.array([2,3,4,5,6,7]),minSize=4)

        t.removeAtIndex(1)
        c = np.array([[2],[4],[5],[6],[7],[0],[0],[0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([2,4,5,6,7]), t.getArray()))
        self.assertEqual(4, t.lastIndex)
        self.assertEqual(5, t.size())

        with self.assertRaises(Exception) as context:
            t.removeAtIndex(-1)
        self.assertTrue('Index out of bounds' in str(context.exception))

        with self.assertRaises(Exception) as context:
            t.removeAtIndex(5)
        self.assertTrue('Index out of bounds' in str(context.exception))

        t.removeAtIndex(1)
        c = np.array([[2], [5], [6], [7]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([2, 5, 6, 7]), t.getArray()))
        self.assertEqual(3, t.lastIndex)
        self.assertEqual(4, t.size())

        t.removeAtIndex(2)
        c = np.array([[2], [5], [7], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([2, 5, 7]), t.getArray()))
        self.assertEqual(2, t.lastIndex)
        self.assertEqual(3, t.size())

        t.removeAtIndex(2)
        c = np.array([[2], [5], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([2, 5]), t.getArray()))
        self.assertEqual(1, t.lastIndex)
        self.assertEqual(2, t.size())

        t.removeAtIndex(0)
        c = np.array([[5], [0], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([5]), t.getArray()))
        self.assertEqual(0, t.lastIndex)
        self.assertEqual(1, t.size())

        t.removeAtIndex(0)
        c = np.array([[0], [0], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(-1, t.lastIndex)
        self.assertEqual(0, t.size())

        t.append(0)
        c = np.array([[0], [0], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([0]), t.getArray()))
        self.assertEqual(0, t.lastIndex)
        self.assertEqual(1, t.size())

        t.append(2)
        c = np.array([[0], [2], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([0,2]), t.getArray()))
        self.assertEqual(1, t.lastIndex)
        self.assertEqual(2, t.size())

    def test2DRemoveAtIndex(self):
        t = TupleArray(np.array([[2,1],[3,2],[4,7],[5,3],[6,0],[9,7]]),minSize=4)

        t.removeAtIndex(1)
        c = np.array([[2,1],[4,7],[5,3],[6,0],[9,7],[0,0],[0,0],[0,0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[2,1],[4,7],[5,3],[6,0],[9,7]]), t.getArray()))
        self.assertEqual(4, t.lastIndex)
        self.assertEqual(10, t.size())

        with self.assertRaises(Exception) as context:
            t.removeAtIndex(-1)
        self.assertTrue('Index out of bounds' in str(context.exception))

        with self.assertRaises(Exception) as context:
            t.removeAtIndex(5)
        self.assertTrue('Index out of bounds' in str(context.exception))

        t.removeAtIndex(1)
        c = np.array([[2,1], [5,3], [6,0], [9,7]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[2,1], [5,3], [6,0], [9,7]]), t.getArray()))
        self.assertEqual(3, t.lastIndex)
        self.assertEqual(8, t.size())

        t.removeAtIndex(2)
        c = np.array([[2,1], [5,3], [9,7], [0,0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[2,1], [5,3], [9,7]]), t.getArray()))
        self.assertEqual(2, t.lastIndex)
        self.assertEqual(6, t.size())

        t.removeAtIndex(2)
        c = np.array([[2,1], [5,3], [0,0], [0,0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[2,1], [5,3]]), t.getArray()))
        self.assertEqual(1, t.lastIndex)
        self.assertEqual(4, t.size())

        t.removeAtIndex(0)
        c = np.array([[5,3], [0,0], [0,0], [0,0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[5,3]]), t.getArray()))
        self.assertEqual(0, t.lastIndex)
        self.assertEqual(2, t.size())

        t.removeAtIndex(0)
        c = np.array([[0,0], [0,0], [0,0], [0,0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(-1, t.lastIndex)
        self.assertEqual(0, t.size())

        t.append([0,0])
        c = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[0,0]]), t.getArray()))
        self.assertEqual(0, t.lastIndex)
        self.assertEqual(2, t.size())

        t.append(np.array([3,2]))
        c = np.array([[0, 0], [3, 2], [0, 0], [0, 0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([[0, 0],[3,2]]), t.getArray()))
        self.assertEqual(1, t.lastIndex)
        self.assertEqual(4, t.size())

    def test2DRemoveValue(self):
        t = TupleArray(np.array([[2, 1], [3, 2], [4, 7], [5, 3], [6, 0], [9, 7]]), minSize=4)
        with self.assertRaises(Exception) as context:
            t.removeFirstElementWithValue([2,3])
        self.assertTrue("This method is only useable for 1D arrays" in str(context.exception))

    def test1DRemoveValue(self):
        t = TupleArray(np.array([2,3,4,5,4,7]), minSize=4)

        t.removeFirstElementWithValue(4)
        c = np.array([[2], [3], [5], [4], [7], [0], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([2, 3, 5, 4, 7]), t.getArray()))
        self.assertEqual(4, t.lastIndex)
        self.assertEqual(5, t.size())

        t.removeFirstElementWithValue(4)
        c = np.array([[2], [3], [5], [7]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([2, 3, 5, 7]), t.getArray()))
        self.assertEqual(3, t.lastIndex)
        self.assertEqual(4, t.size())

        t.removeFirstElementWithValue(2)
        c = np.array([[3], [5], [7], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([3, 5, 7]), t.getArray()))
        self.assertEqual(2, t.lastIndex)
        self.assertEqual(3, t.size())

        t.removeFirstElementWithValue(7)
        c = np.array([[3], [5], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([3, 5]), t.getArray()))
        self.assertEqual(1, t.lastIndex)
        self.assertEqual(2, t.size())

        t.removeFirstElementWithValue(3)
        c = np.array([[5], [0], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([5]), t.getArray()))
        self.assertEqual(0, t.lastIndex)
        self.assertEqual(1, t.size())

        t.removeFirstElementWithValue(5)
        c = np.array([[0], [0], [0], [0]])
        self.assertTrue(np.array_equal(c, t.a))
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(-1, t.lastIndex)
        self.assertEqual(0, t.size())











