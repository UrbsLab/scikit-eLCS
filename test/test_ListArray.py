import unittest
import numpy as np
from skeLCS.DynamicNPArray import ListArray

class Arb():
    def __init__(self,num):
        self.num = 5

    def __eq__(self, other):
        if not isinstance(other,Arb):
            return False

        return other.num == self.num

class TestListArray(unittest.TestCase):

    def testEmpty1DInit(self):
        t = ListArray(k=1)
        self.assertEquals([],t.a)
        self.assertTrue(np.array_equal(np.array([]),t.getArray()))
        self.assertEqual(0,t.size())

    def testSmall1DInit(self):
        t = ListArray(np.array([2,3,6,5]))
        c = [[2],[3],[6],[5]]
        self.assertEquals(c,t.a)
        self.assertTrue(np.array_equal(np.array([2,3,6,5]),t.getArray()))
        self.assertEqual(4, t.size())

    def testLarge1DInit(self):
        t = ListArray(np.array([2, 3, 6, 5,4,5,1,2,5]))
        c = [[2], [3], [6], [5], [4], [5], [1], [2], [5]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([2, 3, 6, 5,4,5,1,2,5]), t.getArray()))
        self.assertEqual(9, t.size())

    def testSmall2DInit(self):
        t = ListArray(np.array([[2, 3],[1, 8],[9, 1],[6, 7],[3, 1]]))
        c = [[2, 3],[1, 8],[9, 1],[6, 7],[3, 1]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([[2, 3],[1, 8],[9, 1],[6, 7],[3, 1]]),t.getArray()))

    def testStrange1DInit(self):
        t = ListArray(np.array([[2],[3],[4]]))
        c = [[2],[3],[4]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([[2],[3],[4]]), t.getArray()))

    def testSmall1DObjInit(self):
        t = ListArray(np.array([Arb(2),Arb(3),Arb(6),Arb(5)]))
        c = [[Arb(2)],[Arb(3)],[Arb(6)],[Arb(5)]]
        self.assertEquals(c,t.a)
        self.assertTrue(np.array_equal(np.array([Arb(2),Arb(3),Arb(6),Arb(5)]),t.getArray()))
        self.assertEqual(4, t.size())

    def testSmall2DObjInit(self):
        t = ListArray(np.array([[Arb(2),Arb(1)],[Arb(3),Arb(0)],[Arb(6),Arb(7)],[Arb(5),Arb(70)]]))
        c = [[Arb(2),Arb(1)],[Arb(3),Arb(0)],[Arb(6),Arb(7)],[Arb(5),Arb(70)]]
        self.assertEquals(c,t.a)
        self.assertTrue(np.array_equal(np.array([[Arb(2),Arb(1)],[Arb(3),Arb(0)],[Arb(6),Arb(7)],[Arb(5),Arb(70)]]),t.getArray()))
        self.assertEqual(8, t.size())

    def testInvalidInput(self):
        with self.assertRaises(Exception) as context:
            t = ListArray(3)

        self.assertTrue("Invalid input array: must be ndarray and have dimension <= 2" in str(context.exception))

    def testInvalidInput2(self):
        with self.assertRaises(Exception) as context:
            t = ListArray(np.array([[[2,3],[3,4]],[[4,3],[2,0]]]))

        self.assertTrue("Invalid input array: must be ndarray and have dimension <= 2" in str(context.exception))

    def testInvalidInput4(self):
        with self.assertRaises(Exception) as context:
            t = ListArray(k=-1)

        self.assertTrue("Must specify valid dimension if you init w/ empty array" in str(context.exception))

    def testInvalidInput5(self):
        with self.assertRaises(Exception) as context:
            t = ListArray()

        self.assertTrue("Must specify valid dimension if you init w/ empty array" in str(context.exception))

    def testInvalidInput6(self):
        with self.assertRaises(Exception) as context:
            t = ListArray(np.array([[]]))

        self.assertTrue("Invalid input array: must be ndarray and have dimension <= 2" in str(context.exception))

    def testInvalidInput7(self):
        with self.assertRaises(Exception) as context:
            t = ListArray(np.array([[],[]]))

        self.assertTrue("Invalid input array: must be ndarray and have dimension <= 2" in str(context.exception))

    def testInvalidInput8(self):
        with self.assertRaises(Exception) as context:
            n = np.array([[2,3]])
            n = np.delete(n,0,axis=0)
            t = ListArray(n)

        self.assertTrue("Invalid input array: must be ndarray and have dimension <= 2" in str(context.exception))

    ####################################################################################################################
    def test1DAppendAndRemoveLast(self):
        t = ListArray(k=1)

        t.append(6)
        c = [[6]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([6]),t.getArray()))
        self.assertEqual(1, t.size())

        t.append(8)
        c = [[6], [8]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([6,8]), t.getArray()))
        self.assertEqual(2, t.size())

        t.append(9)
        c = [[6], [8], [9]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([6,8,9]), t.getArray()))
        self.assertEqual(3, t.size())

        t.removeLast()
        c = [[6], [8]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([6, 8]), t.getArray()))
        self.assertEqual(2, t.size())

        t.removeLast()
        c = [[6]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([6]), t.getArray()))
        self.assertEqual(1, t.size())

        t.removeLast()
        c = []
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(0, t.size())

        t.append(20)
        c = [[20]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([20]), t.getArray()))
        self.assertEqual(1, t.size())

    def test2DAppendAndRemoveLast(self):
        t = ListArray(k=2)
        t.append(np.array([6,1]))
        c =[[6,1]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([[6,1]]),t.getArray()))
        self.assertEqual(2, t.size())

        t.append(np.array([8,2]))
        c = [[6,1], [8,2]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([[6,1], [8,2]]), t.getArray()))
        self.assertEqual(4, t.size())

        t.removeLast()
        c = [[6, 1]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([[6, 1]]), t.getArray()))
        self.assertEqual(2, t.size())

        t.removeLast()
        c = []
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(0, t.size())

        t.append(np.array([20,11]))
        c = [[20, 11]]
        self.assertEquals(c, t.a)
        self.assertTrue(np.array_equal(np.array([[20, 11]]), t.getArray()))
        self.assertEquals(2, t.size())

    def test1DRemoveAtIndex(self):
        t = ListArray(np.array([2,3,4,5,6,7]))

        t.removeAtIndex(1)
        c = [[2],[4],[5],[6],[7]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([2,4,5,6,7]), t.getArray()))
        self.assertEqual(5, t.size())

        t.removeAtIndex(1)
        c = [[2], [5], [6], [7]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([2, 5, 6, 7]), t.getArray()))
        self.assertEqual(4, t.size())

        t.removeAtIndex(2)
        c = [[2], [5], [7]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([2, 5, 7]), t.getArray()))
        self.assertEqual(3, t.size())

        t.removeAtIndex(2)
        c = [[2], [5]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([2, 5]), t.getArray()))
        self.assertEqual(2, t.size())

        t.removeAtIndex(0)
        c = [[5]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([5]), t.getArray()))
        self.assertEqual(1, t.size())

        t.removeAtIndex(0)
        c = []
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(0, t.size())

        t.append(0)
        c = [[0]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([0]), t.getArray()))
        self.assertEqual(1, t.size())

        t.append(2)
        c = [[0], [2]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([0,2]), t.getArray()))
        self.assertEqual(2, t.size())

    def test2DRemoveAtIndex(self):
        t = ListArray(np.array([[2,1],[3,2],[4,7],[5,3],[6,0],[9,7]]))

        t.removeAtIndex(1)
        c = [[2,1],[4,7],[5,3],[6,0],[9,7]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([[2,1],[4,7],[5,3],[6,0],[9,7]]), t.getArray()))
        self.assertEqual(10, t.size())

        t.removeAtIndex(1)
        c = [[2,1], [5,3], [6,0], [9,7]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([[2,1], [5,3], [6,0], [9,7]]), t.getArray()))
        self.assertEqual(8, t.size())

        t.removeAtIndex(2)
        c = [[2,1], [5,3], [9,7]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([[2,1], [5,3], [9,7]]), t.getArray()))
        self.assertEqual(6, t.size())

        t.removeAtIndex(2)
        c = [[2,1], [5,3]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([[2,1], [5,3]]), t.getArray()))
        self.assertEqual(4, t.size())

        t.removeAtIndex(0)
        c = [[5,3]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([[5,3]]), t.getArray()))
        self.assertEqual(2, t.size())

        t.removeAtIndex(0)
        c = []
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(0, t.size())

        t.append(np.array([0,0]))
        c = [[0, 0]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([[0,0]]), t.getArray()))
        self.assertEqual(2, t.size())

        t.append(np.array([3,2]))
        c = [[0, 0], [3, 2]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([[0, 0],[3,2]]), t.getArray()))
        self.assertEqual(4, t.size())

    def test2DRemoveValue(self):
        t = ListArray(np.array([[2, 1], [3, 2], [4, 7], [5, 3], [6, 0], [9, 7]]))
        with self.assertRaises(Exception) as context:
            t.removeFirstElementWithValue([2,3])
        self.assertTrue("This method is only useable for 1D arrays" in str(context.exception))

    def test1DRemoveValue(self):
        t = ListArray(np.array([2,3,4,5,4,7]))

        t.removeFirstElementWithValue(4)
        c = [[2], [3], [5], [4], [7]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([2, 3, 5, 4, 7]), t.getArray()))
        self.assertEqual(5, t.size())

        t.removeFirstElementWithValue(4)
        c = [[2], [3], [5], [7]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([2, 3, 5, 7]), t.getArray()))
        self.assertEqual(4, t.size())

        t.removeFirstElementWithValue(2)
        c = [[3], [5], [7]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([3, 5, 7]), t.getArray()))
        self.assertEqual(3, t.size())

        t.removeFirstElementWithValue(7)
        c = [[3], [5]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([3, 5]), t.getArray()))
        self.assertEqual(2, t.size())

        t.removeFirstElementWithValue(3)
        c = [[5]]
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([5]), t.getArray()))
        self.assertEqual(1, t.size())

        t.removeFirstElementWithValue(5)
        c = []
        self.assertEqual(c, t.a)
        self.assertTrue(np.array_equal(np.array([]), t.getArray()))
        self.assertEqual(0, t.size())











