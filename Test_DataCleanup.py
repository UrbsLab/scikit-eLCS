import unittest
import DataCleanup
import pandas as pd
import numpy as np

class TestDataCleanup(unittest.TestCase):

    def testInitMissingData(self):
        # Tests if init filters missing data into NAs
        se = DataCleanup.StringEnumerator("DataSets/Tests/MissingFeatureData.csv","phenotype")
        cFeatures = np.array([["1.0","NA","1.0","4.0"],["2.0","0.0","1.0","NA"],["4.0","NA","1.0","2.0"],["NA","1.0","NA","1.0"],["6.0","NA","1.0","1.0"]])
        self.assertTrue(np.array_equal(cFeatures,se.dataFeatures))

    def testInitHeaders(self):
        # Tests if init gets the headers correct
        se = DataCleanup.StringEnumerator("DataSets/Tests/MissingFeatureData.csv", "phenotype")
        cHeaders = np.array(["N1","N2","N3","N4"])
        self.assertTrue(np.array_equal(cHeaders, se.dataHeaders))

    def testInitFeaturesAndClass(self):
        # Tests if init gets the features and class arrays correct
        se = DataCleanup.StringEnumerator("DataSets/Tests/MissingFeatureData.csv", "phenotype")
        cFeatures = np.array([["1.0", "NA", "1.0", "4.0"], ["2.0", "0.0", "1.0", "NA"], ["4.0", "NA", "1.0", "2.0"], ["NA", "1.0", "NA", "1.0"],["6.0", "NA", "1.0", "1.0"]])
        cClasses = np.array(["1", "0", "1", "0", "1"])
        self.assertTrue(np.array_equal(cFeatures, se.dataFeatures))
        self.assertTrue(np.array_equal(cClasses, se.dataPhenotypes))

    def testInitFeaturesAndClassRemoval(self):
        # Tests if init gets the features and class arrays correct given missing phenotype data
        se = DataCleanup.StringEnumerator("DataSets/Tests/MissingFeatureAndPhenotypeData.csv", "phenotype")
        cFeatures = np.array([["1.0", "NA", "1.0", "4.0"], ["NA", "1.0", "NA", "1.0"], ["6.0", "NA", "1.0", "1.0"]])
        cClasses = np.array(["1.0", "0.0", "1.0"])
        self.assertTrue(np.array_equal(cFeatures, se.dataFeatures))
        self.assertTrue(np.array_equal(cClasses, se.dataPhenotypes))

    def testChangeClassAndHeaderNames(self):
        # Changes header and class names. Checks map, and classLabel/dataHeaders correctness
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        se.changeClassName("country")
        se.changeHeaderName("N1","gender")
        se.changeHeaderName("N2","N1")
        se.changeHeaderName("N1","floats")
        se.changeHeaderName("N3","phenotype")
        se.changeHeaderName("phenotype","age")
        cHeaders = np.array(["gender","floats","age"])
        self.assertTrue(np.array_equal(cHeaders,se.dataHeaders))
        self.assertTrue(np.array_equal("country", se.classLabel))

    def testChangeClassAndHeaderNames2(self):
        # Changes header and class names. Checks map, and classLabel/dataHeaders correctness
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        se.addClassConverterRandom()
        se.changeHeaderName("N1","gender")
        se.addAttributeConverterRandom("gender")
        se.changeHeaderName("gender","Gender")
        se.addAttributeConverterRandom("Gender")
        se.addAttributeConverterRandom("Gender")
        se.addAttributeConverterRandom("gender")
        se.addAttributeConverterRandom("N3")
        se.changeHeaderName("N3","Age")

        cHeaders = np.array(["Gender","N2","Age"])
        cMap = {"phenotype":{"china":"0","japan":"1","russia":"2"},"Gender":{"male":"0","female":"1"},"Age":{"young":"0","old":"1"}}
        self.assertTrue(np.array_equal(cHeaders,se.dataHeaders))
        self.assertTrue(np.array_equal("phenotype", se.classLabel))
        self.assertTrue(se.map == cMap)

    def testChangeClassNameInvalid(self):
        # Changes class name to an existing header name should raise exception
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        with self.assertRaises(Exception) as context:
            se.changeClassName("N1")

        self.assertTrue("New Class Name Cannot Be An Already Existing Data Header Name" in str(context.exception))


    def testChangeHeaderNameInvalid(self):
        # Changes header name to an existing header or class name should raise exception
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        with self.assertRaises(Exception) as context:
            se.changeHeaderName("N1","N2")

        self.assertTrue("New Class Name Cannot Be An Already Existing Data Header or Phenotype Name" in str(context.exception))

    def testChangeHeaderNameInvalid2(self):
        # Changes non existing header name should raise exception
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        with self.assertRaises(Exception) as context:
            se.changeHeaderName("N", "N5")
        self.assertTrue("Current Header Doesn't Exist" in str(context.exception))

    def testDeleteAttribute(self):
        # Deletes attributes and checks map, headers, and arrays for correctness
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        se.changeHeaderName("N1","gender")
        se.addAttributeConverterRandom("gender")
        se.addAttributeConverterRandom("N3")
        se.deleteAttribute("gender")
        cHeaders = np.array(["N2","N3"])
        cMap = {"N3": {"young": "0", "old": "1"}}
        self.assertTrue(np.array_equal(cHeaders, se.dataHeaders))
        self.assertTrue(np.array_equal("phenotype", se.classLabel))
        self.assertTrue(se.map == cMap)

    def testDeleteNonexistentAttribute(self):
        # Deletes nonexistent attribute
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        with self.assertRaises(Exception) as context:
            se.deleteAttribute("N")
        self.assertTrue("Header Doesn't Exist" in str(context.exception))

    def testDeleteInstancesWithMissing(self):
        # Deletes instances and checks arrays for correctness
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        se.changeHeaderName("N1","gender")
        se.addAttributeConverterRandom("gender")
        se.addAttributeConverterRandom("N3")
        se.addClassConverterRandom()
        se.convertAllAttributes()
        se.deleteAllInstancesWithoutHeaderData("gender")
        se.deleteAllInstancesWithoutHeaderData("N2")
        se.deleteAllInstancesWithoutHeaderData("N3")
        cHeaders = np.array(["gender","N2","N3"])
        cMap = {"phenotype":{"china":"0","japan":"1","russia":"2"},"gender":{"male":"0","female":"1"},"N3":{"young":"0","old":"1"}}
        cArray = np.array([["0","1.2","0"],["1","-0.4","1"]])
        cPArray = np.array(["0","0"])
        self.assertTrue(np.array_equal(cHeaders, se.dataHeaders))
        self.assertTrue(np.array_equal("phenotype", se.classLabel))
        self.assertTrue(np.array_equal(cArray, se.dataFeatures))
        self.assertTrue(np.array_equal(cPArray, se.dataPhenotypes))
        self.assertTrue(se.map == cMap)

    def testDeleteInstancesWithMissing2(self):
        # Deletes instances and checks arrays for correctness
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        se.changeHeaderName("N1","gender")
        se.deleteAllInstancesWithoutHeaderData("gender")
        se.deleteAllInstancesWithoutHeaderData("N2")
        se.deleteAllInstancesWithoutHeaderData("N3")

        se.addAttributeConverterRandom("gender")
        se.addAttributeConverterRandom("N3")
        se.addClassConverterRandom()
        se.convertAllAttributes()

        cHeaders = np.array(["gender","N2","N3"])
        cMap = {"phenotype":{"china":"0"},"gender":{"male":"0","female":"1"},"N3":{"young":"0","old":"1"}}
        cArray = np.array([["0","1.2","0"],["1","-0.4","1"]])
        cPArray = np.array(["0","0"])
        self.assertTrue(np.array_equal(cHeaders, se.dataHeaders))
        self.assertTrue(np.array_equal("phenotype", se.classLabel))
        self.assertTrue(np.array_equal(cArray, se.dataFeatures))
        self.assertTrue(np.array_equal(cPArray, se.dataPhenotypes))
        self.assertTrue(se.map == cMap)

    def testNumericCheck(self):
        # Checks non missing numeric
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        self.assertFalse(se.checkIsFullNumeric())
        se.addAttributeConverterRandom("N1")
        se.convertAllAttributes()
        self.assertFalse(se.checkIsFullNumeric())
        se.addAttributeConverterRandom("N3")
        se.addClassConverterRandom()
        se.convertAllAttributes()
        self.assertTrue(se.checkIsFullNumeric())

        se2 = DataCleanup.StringEnumerator("DataSets/Tests/MissingFeatureData.csv", "phenotype")
        self.assertTrue(se2.checkIsFullNumeric())

    def testGetParamsFail(self):
        # Get params when not all features/class have been enumerated
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        with self.assertRaises(Exception) as context:
            se.getParams()
        self.assertTrue("Features and Phenotypes must be fully numeric" in str(context.exception))

    def testGetParams1(self):
        # Get Params Test
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        se.changeHeaderName("N1","gender")
        se.changeHeaderName("N2","floats")
        se.changeHeaderName("N3","age")
        se.changeClassName("country")
        se.addAttributeConverterRandom("gender")
        se.addAttributeConverterRandom("age")
        #se.addAttributeConverterRandom("floats") #You can convert "floats" to discrete values as well
        se.addClassConverterRandom()
        se.convertAllAttributes()
        dataHeaders,classLabel,dataFeatures,dataPhenotypes = se.getParams()
        cHeaders = np.array(["gender","floats","age"])
        cFeatures = np.array([[0,1.2,0],[1,0.3,np.nan],[1,-0.4,1],[np.nan,0,0]])
        cPhenotypes = np.array([0,1,0,2])
        self.assertEqual("country",classLabel)
        self.assertTrue(np.array_equal(cHeaders,dataHeaders))
        self.assertTrue(np.allclose(cFeatures,dataFeatures,equal_nan=True))
        self.assertTrue(np.allclose(cPhenotypes, dataPhenotypes, equal_nan=True))

    def testGetParams2(self):
        # Get Params Test
        se = DataCleanup.StringEnumerator("DataSets/Tests/StringData.csv", "phenotype")
        se.changeHeaderName("N1", "gender")
        se.changeHeaderName("N2", "floats")
        se.changeHeaderName("N3", "age")
        se.changeClassName("country")
        se.addAttributeConverter("gender",np.array(["female","male","NA","other"]))
        se.addAttributeConverter("age",np.array(["old","young"]))
        se.addClassConverterRandom()
        se.convertAllAttributes()
        dataHeaders, classLabel, dataFeatures, dataPhenotypes = se.getParams()
        cHeaders = np.array(["gender", "floats", "age"])
        cFeatures = np.array([[1, 1.2, 1], [0, 0.3, np.nan], [0, -0.4, 0], [np.nan, 0, 1]])
        cPhenotypes = np.array([0, 1, 0, 2])
        self.assertEqual("country", classLabel)
        self.assertTrue(np.array_equal(cHeaders, dataHeaders))
        self.assertTrue(np.allclose(cFeatures, dataFeatures, equal_nan=True))
        self.assertTrue(np.allclose(cPhenotypes, dataPhenotypes, equal_nan=True))