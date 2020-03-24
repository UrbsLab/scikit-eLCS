import unittest

import Test_DataCleanup
import Test_eLCS


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromModule(Test_DataCleanup))
    suite.addTests(loader.loadTestsFromModule(Test_eLCS))

    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)

    print(result)