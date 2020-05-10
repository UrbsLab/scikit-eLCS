import unittest
import skeLCS.DataCleanup as DataCleanup
import pandas as pd
import numpy as np
import os

THIS_DIR = os.path.dirname(os.path.abspath("test_eLCS.py"))
if THIS_DIR[-4:] == "test": #Patch that ensures testing from Scikit not test directory
    THIS_DIR = THIS_DIR[:-5]

class TestDataCleanup(unittest.TestCase):

    def testInitMissingData(self):
        # Tests if init filters missing data into NAs
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/MissingFeatureData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        cFeatures = np.array([["1.0","NA","1.0","4.0"],["2.0","0.0","1.0","NA"],["4.0","NA","1.0","2.0"],["NA","1.0","NA","1.0"],["6.0","NA","1.0","1.0"]])
        self.assertTrue(np.array_equal(cFeatures,se.dataFeatures))

    def testInitHeaders(self):
        # Tests if init gets the headers correct
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/MissingFeatureData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        cHeaders = np.array(["N1","N2","N3","N4"])
        self.assertTrue(np.array_equal(cHeaders, se.dataHeaders))

    def testInitFeaturesAndClass(self):
        # Tests if init gets the features and class arrays correct
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/MissingFeatureData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        cFeatures = np.array([["1.0", "NA", "1.0", "4.0"], ["2.0", "0.0", "1.0", "NA"], ["4.0", "NA", "1.0", "2.0"], ["NA", "1.0", "NA", "1.0"],["6.0", "NA", "1.0", "1.0"]])
        cClasses = np.array(["1", "0", "1", "0", "1"])
        self.assertTrue(np.array_equal(cFeatures, se.dataFeatures))
        self.assertTrue(np.array_equal(cClasses, se.dataPhenotypes))

    def testInitFeaturesAndClassRemoval(self):
        # Tests if init gets the features and class arrays correct given missing phenotype data
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/MissingFeatureAndPhenotypeData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        cFeatures = np.array([["1.0", "NA", "1.0", "4.0"], ["NA", "1.0", "NA", "1.0"], ["6.0", "NA", "1.0", "1.0"]])
        cClasses = np.array(["1.0", "0.0", "1.0"])
        self.assertTrue(np.array_equal(cFeatures, se.dataFeatures))
        self.assertTrue(np.array_equal(cClasses, se.dataPhenotypes))

    def testChangeClassAndHeaderNames(self):
        # Changes header and class names. Checks map, and classLabel/dataHeaders correctness
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        se.change_class_name("country")
        se.change_header_name("N1","gender")
        se.change_header_name("N2","N1")
        se.change_header_name("N1","floats")
        se.change_header_name("N3","phenotype")
        se.change_header_name("phenotype","age")
        cHeaders = np.array(["gender","floats","age"])
        self.assertTrue(np.array_equal(cHeaders,se.dataHeaders))
        self.assertTrue(np.array_equal("country", se.classLabel))

    def testChangeClassAndHeaderNames2(self):
        # Changes header and class names. Checks map, and classLabel/dataHeaders correctness
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        se.add_class_converter_random()
        se.change_header_name("N1","gender")
        se.add_attribute_converter_random("gender")
        se.change_header_name("gender","Gender")
        se.add_attribute_converter_random("Gender")
        se.add_attribute_converter_random("Gender")
        se.add_attribute_converter_random("gender")
        se.add_attribute_converter_random("N3")
        se.change_header_name("N3","Age")

        cHeaders = np.array(["Gender","N2","Age"])
        cMap = {"phenotype":{"china":"0","japan":"1","russia":"2"},"Gender":{"male":"0","female":"1"},"Age":{"young":"0","old":"1"}}
        self.assertTrue(np.array_equal(cHeaders,se.dataHeaders))
        self.assertTrue(np.array_equal("phenotype", se.classLabel))
        self.assertTrue(se.map == cMap)

    def testChangeClassNameInvalid(self):
        # Changes class name to an existing header name should raise exception
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        with self.assertRaises(Exception) as context:
            se.change_class_name("N1")

        self.assertTrue("New Class Name Cannot Be An Already Existing Data Header Name" in str(context.exception))


    def testChangeHeaderNameInvalid(self):
        # Changes header name to an existing header or class name should raise exception
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        with self.assertRaises(Exception) as context:
            se.change_header_name("N1","N2")

        self.assertTrue("New Class Name Cannot Be An Already Existing Data Header or Phenotype Name" in str(context.exception))

    def testChangeHeaderNameInvalid2(self):
        # Changes non existing header name should raise exception
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        with self.assertRaises(Exception) as context:
            se.change_header_name("N", "N5")
        self.assertTrue("Current Header Doesn't Exist" in str(context.exception))

    def testDeleteAttribute(self):
        # Deletes attributes and checks map, headers, and arrays for correctness
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        se.change_header_name("N1","gender")
        se.add_attribute_converter_random("gender")
        se.add_attribute_converter_random("N3")
        se.delete_attribute("gender")
        cHeaders = np.array(["N2","N3"])
        cMap = {"N3": {"young": "0", "old": "1"}}
        self.assertTrue(np.array_equal(cHeaders, se.dataHeaders))
        self.assertTrue(np.array_equal("phenotype", se.classLabel))
        self.assertTrue(se.map == cMap)

    def testDeleteNonexistentAttribute(self):
        # Deletes nonexistent attribute
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        with self.assertRaises(Exception) as context:
            se.delete_attribute("N")
        self.assertTrue("Header Doesn't Exist" in str(context.exception))

    def testDeleteInstancesWithMissing(self):
        # Deletes instances and checks arrays for correctness
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        se.change_header_name("N1","gender")
        se.add_attribute_converter_random("gender")
        se.add_attribute_converter_random("N3")
        se.add_class_converter_random()
        se.convert_all_attributes()
        se.delete_all_instances_without_header_data("gender")
        se.delete_all_instances_without_header_data("N2")
        se.delete_all_instances_without_header_data("N3")
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
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        se.change_header_name("N1","gender")
        se.delete_all_instances_without_header_data("gender")
        se.delete_all_instances_without_header_data("N2")
        se.delete_all_instances_without_header_data("N3")

        se.add_attribute_converter_random("gender")
        se.add_attribute_converter_random("N3")
        se.add_class_converter_random()
        se.convert_all_attributes()

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
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        self.assertFalse(se.check_is_full_numeric())
        se.add_attribute_converter_random("N1")
        se.convert_all_attributes()
        self.assertFalse(se.check_is_full_numeric())
        se.add_attribute_converter_random("N3")
        se.add_class_converter_random()
        se.convert_all_attributes()
        self.assertTrue(se.check_is_full_numeric())

        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/MissingFeatureData.csv")
        se2 = DataCleanup.StringEnumerator(dataPath, "phenotype")
        self.assertTrue(se2.check_is_full_numeric())

    def testGetParamsFail(self):
        # Get params when not all features/class have been enumerated
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        with self.assertRaises(Exception) as context:
            se.get_params()
        self.assertTrue("Features and Phenotypes must be fully numeric" in str(context.exception))

    def testGetParams1(self):
        # Get Params Test
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        se.change_header_name("N1","gender")
        se.change_header_name("N2","floats")
        se.change_header_name("N3","age")
        se.change_class_name("country")
        se.add_attribute_converter_random("gender")
        se.add_attribute_converter_random("age")
        #se.add_attribute_converter_random("floats") #You can convert "floats" to discrete values as well
        se.add_class_converter_random()
        se.convert_all_attributes()
        dataHeaders,classLabel,dataFeatures,dataPhenotypes = se.get_params()
        cHeaders = np.array(["gender","floats","age"])
        cFeatures = np.array([[0,1.2,0],[1,0.3,np.nan],[1,-0.4,1],[np.nan,0,0]])
        cPhenotypes = np.array([0,1,0,2])
        self.assertEqual("country",classLabel)
        self.assertTrue(np.array_equal(cHeaders,dataHeaders))
        self.assertTrue(np.allclose(cFeatures,dataFeatures,equal_nan=True))
        self.assertTrue(np.allclose(cPhenotypes, dataPhenotypes, equal_nan=True))

    def testGetParams2(self):
        # Get Params Test
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData.csv")
        se = DataCleanup.StringEnumerator(dataPath, "phenotype")
        se.change_header_name("N1", "gender")
        se.change_header_name("N2", "floats")
        se.change_header_name("N3", "age")
        se.change_class_name("country")
        se.add_attribute_converter("gender",np.array(["female","male","NA","other"]))
        se.add_attribute_converter("age",np.array(["old","young"]))
        se.add_class_converter_random()
        se.convert_all_attributes()
        dataHeaders, classLabel, dataFeatures, dataPhenotypes = se.get_params()
        cHeaders = np.array(["gender", "floats", "age"])
        cFeatures = np.array([[1, 1.2, 1], [0, 0.3, np.nan], [0, -0.4, 0], [np.nan, 0, 1]])
        cPhenotypes = np.array([0, 1, 0, 2])
        self.assertEqual("country", classLabel)
        self.assertTrue(np.array_equal(cHeaders, dataHeaders))
        self.assertTrue(np.allclose(cFeatures, dataFeatures, equal_nan=True))
        self.assertTrue(np.allclose(cPhenotypes, dataPhenotypes, equal_nan=True))
    #
    # def testPrintInvalids(self):
    #     dataPath = os.path.join(THIS_DIR, "test/DataSets/Tests/StringData2.csv")
    #     se = DataCleanup.StringEnumerator(dataPath, "phenotype")
    #     se.print_invalid_attributes()