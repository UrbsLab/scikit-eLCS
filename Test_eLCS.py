import unittest
import DataCleanup
import pandas as pd
import numpy as np
from eLCS import *


class Test_eLCS(unittest.TestCase):

    '''SECTION 1: TEST eLCS Parameters
    '''
    def testParamLearningIterationsNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(learningIterations="hello")
        self.assertTrue("learningIterations param must be nonnegative integer" in str(context.exception))

    def testParamLearningIterationsInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(learningIterations=3.3)
        self.assertTrue("learningIterations param must be nonnegative integer" in str(context.exception))

    def testParamLearningIterationsInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(learningIterations=-2)
        self.assertTrue("learningIterations param must be nonnegative integer" in str(context.exception))

    def testParamLearningIterations(self):
        clf = eLCS(learningIterations=2000)
        self.assertEqual(clf.learningIterations,2000)


    def testParamTrackingFrequencyNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(trackingFrequency="hello")
        self.assertTrue("trackingFrequency param must be nonnegative integer" in str(context.exception))

    def testParamTrackingFrequencyInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(trackingFrequency=3.3)
        self.assertTrue("trackingFrequency param must be nonnegative integer" in str(context.exception))

    def testParamTrackingFrequencyInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(trackingFrequency=-2)
        self.assertTrue("trackingFrequency param must be nonnegative integer" in str(context.exception))

    def testParamTrackFrequency(self):
        clf = eLCS(trackingFrequency=200)
        self.assertEqual(clf.trackingFrequency,200)


    def testParamLearningCheckpointsNonarray(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(learningCheckpoints=2)
        self.assertTrue("learningCheckpoints param must be ndarray" in str(context.exception))

    def testParamLearningCheckpointsNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(learningCheckpoints=np.array([2,100,"hi",200]))
        self.assertTrue("All learningCheckpoints elements param must be nonnegative integers" in str(context.exception))

    def testParamLearningCheckpointsInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(learningCheckpoints=np.array([2,100,200.2,200]))
        self.assertTrue("All learningCheckpoints elements param must be nonnegative integers" in str(context.exception))

    def testParamLearningCheckpointsInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(learningCheckpoints=np.array([2,100,-200,200]))
        self.assertTrue("All learningCheckpoints elements param must be nonnegative integers" in str(context.exception))

    def testParamLearningCheckpoints(self):
        clf = eLCS(learningCheckpoints=np.array([2, 100, 200, 300]))
        self.assertTrue(np.array_equal(clf.learningCheckpoints,np.array([2, 100, 200, 300])))


    def testEvalWhileFitInvalid(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(evalWhileFit=2)
        self.assertTrue("evalWhileFit param must be boolean" in str(context.exception))

    def testEvalWhileFit(self):
        clf = eLCS(evalWhileFit=True)
        self.assertEqual(clf.evalWhileFit,True)


    def testParamNNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(N="hello")
        self.assertTrue("N param must be nonnegative integer" in str(context.exception))

    def testParamNInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(N=3.3)
        self.assertTrue("N param must be nonnegative integer" in str(context.exception))

    def testParamNInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(N=-2)
        self.assertTrue("N param must be nonnegative integer" in str(context.exception))

    def testParamN(self):
        clf = eLCS(N=2000)
        self.assertEqual(clf.N,2000)


    def testParamP_SpecInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(p_spec="hello")
        self.assertTrue("p_spec param must be float from 0 - 1" in str(context.exception))

    def testParamP_SpecInv2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(p_spec=3)
        self.assertTrue("p_spec param must be float from 0 - 1" in str(context.exception))

    def testParamP_SpecInv3(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(p_spec=-1.2)
        self.assertTrue("p_spec param must be float from 0 - 1" in str(context.exception))

    def testParamP_Spec1(self):
        clf = eLCS(p_spec=0)
        self.assertEqual(clf.p_spec,0)

    def testParamP_Spec2(self):
        clf = eLCS(p_spec=0.3)
        self.assertEqual(clf.p_spec,0.3)

    def testParamP_Spec3(self):
        clf = eLCS(p_spec=1)
        self.assertEqual(clf.p_spec,1)


    def testDiscreteAttributeLimitInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(discreteAttributeLimit="h")
        self.assertTrue("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscreteAttributeLimitInv2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(discreteAttributeLimit=-10)
        self.assertTrue("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscreteAttributeLimitInv3(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(discreteAttributeLimit=1.2)
        self.assertTrue("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscreteAttributeLimit1(self):
        clf = eLCS(discreteAttributeLimit=10)
        self.assertEqual(clf.discreteAttributeLimit,10)

    def testDiscreteAttributeLimit2(self):
        clf = eLCS(discreteAttributeLimit="c")
        self.assertEqual(clf.discreteAttributeLimit,"c")

    def testDiscreteAttributeLimit3(self):
        clf = eLCS(discreteAttributeLimit="d")
        self.assertEqual(clf.discreteAttributeLimit,"d")


    def testParamSpecAttrNonarray(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(specifiedAttributes=2)
        self.assertTrue("specifiedAttributes param must be ndarray" in str(context.exception))

    def testParamSpecAttrNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(specifiedAttributes=np.array([2,100,"hi",200]))
        self.assertTrue("All specifiedAttributes elements param must be nonnegative integers" in str(context.exception))

    def testParamSpecAttrInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(specifiedAttributes=np.array([2,100,200.2,200]))
        self.assertTrue("All specifiedAttributes elements param must be nonnegative integers" in str(context.exception))

    def testParamSpecAttrInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(specifiedAttributes=np.array([2,100,-200,200]))
        self.assertTrue("All specifiedAttributes elements param must be nonnegative integers" in str(context.exception))

    def testParamSpecAttr(self):
        clf = eLCS(specifiedAttributes=np.array([2, 100, 200, 300]))
        self.assertTrue(np.array_equal(clf.specifiedAttributes,np.array([2, 100, 200, 300])))


    def testDiscretePhenotypeLimitInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(discretePhenotypeLimit="h")
        self.assertTrue(
            "discretePhenotypeLimit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscretePhenotypeLimitInv2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(discretePhenotypeLimit=-10)
        self.assertTrue(
            "discretePhenotypeLimit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscretePhenotypeLimitInv3(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(discretePhenotypeLimit=1.2)
        self.assertTrue(
            "discretePhenotypeLimit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscretePhenotypeLimit1(self):
        clf = eLCS(discretePhenotypeLimit=10)
        self.assertEqual(clf.discretePhenotypeLimit, 10)

    def testDiscretePhenotypeLimit2(self):
        clf = eLCS(discretePhenotypeLimit="c")
        self.assertEqual(clf.discretePhenotypeLimit, "c")

    def testDiscretePhenotypeLimit3(self):
        clf = eLCS(discretePhenotypeLimit="d")
        self.assertEqual(clf.discretePhenotypeLimit, "d")


    def testNuInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(nu="hi")
        self.assertTrue("nu param must be float" in str(context.exception))

    def testNu1(self):
        clf = eLCS(nu = -1)
        self.assertEqual(clf.nu,-1)

    def testNu2(self):
        clf = eLCS(nu = 3)
        self.assertEqual(clf.nu,3)

    def testNu3(self):
        clf = eLCS(nu = 1.2)
        self.assertEqual(clf.nu,1.2)


    def testBetaInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(beta="hi")
        self.assertTrue("beta param must be float" in str(context.exception))

    def testBeta1(self):
        clf = eLCS(beta = -1)
        self.assertEqual(clf.beta,-1)

    def testBeta2(self):
        clf = eLCS(beta = 3)
        self.assertEqual(clf.beta,3)

    def testBeta3(self):
        clf = eLCS(beta = 1.2)
        self.assertEqual(clf.beta,1.2)


    def testDeltaInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(delta="hi")
        self.assertTrue("delta param must be float" in str(context.exception))

    def testDelta1(self):
        clf = eLCS(delta = -1)
        self.assertEqual(clf.delta,-1)

    def testDelta2(self):
        clf = eLCS(delta = 3)
        self.assertEqual(clf.delta,3)

    def testDelta3(self):
        clf = eLCS(delta = 1.2)
        self.assertEqual(clf.delta,1.2)


    def testInitFitInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(init_fit="hi")
        self.assertTrue("init_fit param must be float" in str(context.exception))

    def testInitFit1(self):
        clf = eLCS(init_fit = -1)
        self.assertEqual(clf.init_fit,-1)

    def testInitFit2(self):
        clf = eLCS(init_fit = 3)
        self.assertEqual(clf.init_fit,3)

    def testInitFit3(self):
        clf = eLCS(init_fit = 1.2)
        self.assertEqual(clf.init_fit,1.2)


    def testFitnessReductionInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(fitnessReduction="hi")
        self.assertTrue("fitnessReduction param must be float" in str(context.exception))

    def testFitnessReduction1(self):
        clf = eLCS(fitnessReduction = -1)
        self.assertEqual(clf.fitnessReduction,-1)

    def testFitnessReduction2(self):
        clf = eLCS(fitnessReduction = 3)
        self.assertEqual(clf.fitnessReduction,3)

    def testFitnessReduction3(self):
        clf = eLCS(fitnessReduction = 1.2)
        self.assertEqual(clf.fitnessReduction,1.2)


    def testParamChiInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(chi="hello")
        self.assertTrue("chi param must be float from 0 - 1" in str(context.exception))

    def testParamChiInv2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(chi=3)
        self.assertTrue("chi param must be float from 0 - 1" in str(context.exception))

    def testParamChiInv3(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(chi=-1.2)
        self.assertTrue("chi param must be float from 0 - 1" in str(context.exception))

    def testParamChi1(self):
        clf = eLCS(chi=0)
        self.assertEqual(clf.chi,0)

    def testParamChi2(self):
        clf = eLCS(chi=0.3)
        self.assertEqual(clf.chi,0.3)

    def testParamChi3(self):
        clf = eLCS(chi=1)
        self.assertEqual(clf.chi,1)


    def testParamUpsilonInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(upsilon="hello")
        self.assertTrue("upsilon param must be float from 0 - 1" in str(context.exception))

    def testParamUpsilonInv2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(upsilon=3)
        self.assertTrue("upsilon param must be float from 0 - 1" in str(context.exception))

    def testParamUpsilonInv3(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(upsilon=-1.2)
        self.assertTrue("upsilon param must be float from 0 - 1" in str(context.exception))

    def testParamUpsilon1(self):
        clf = eLCS(upsilon=0)
        self.assertEqual(clf.upsilon,0)

    def testParamUpsilon2(self):
        clf = eLCS(upsilon=0.3)
        self.assertEqual(clf.upsilon,0.3)

    def testParamUpsilon3(self):
        clf = eLCS(upsilon=1)
        self.assertEqual(clf.upsilon,1)


    def testParamAccSubInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(acc_sub="hello")
        self.assertTrue("acc_sub param must be float from 0 - 1" in str(context.exception))

    def testParamAccSubInv2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(acc_sub=3)
        self.assertTrue("acc_sub param must be float from 0 - 1" in str(context.exception))

    def testParamAccSubInv3(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(acc_sub=-1.2)
        self.assertTrue("acc_sub param must be float from 0 - 1" in str(context.exception))

    def testParamAccSub1(self):
        clf = eLCS(acc_sub=0)
        self.assertEqual(clf.acc_sub,0)

    def testParamAccSub2(self):
        clf = eLCS(acc_sub=0.3)
        self.assertEqual(clf.acc_sub,0.3)

    def testParamAccSub3(self):
        clf = eLCS(acc_sub=1)
        self.assertEqual(clf.acc_sub,1)


    def testParamThetaSelInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_sel="hello")
        self.assertTrue("theta_sel param must be float from 0 - 1" in str(context.exception))

    def testParamThetaSelInv2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_sel=3)
        self.assertTrue("theta_sel param must be float from 0 - 1" in str(context.exception))

    def testParamThetaSelInv3(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_sel=-1.2)
        self.assertTrue("theta_sel param must be float from 0 - 1" in str(context.exception))

    def testParamThetaSel1(self):
        clf = eLCS(theta_sel=0)
        self.assertEqual(clf.theta_sel,0)

    def testParamThetaSel2(self):
        clf = eLCS(theta_sel=0.3)
        self.assertEqual(clf.theta_sel,0.3)

    def testParamThetaSel3(self):
        clf = eLCS(theta_sel=1)
        self.assertEqual(clf.theta_sel,1)


    def testParamThetaGAInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_GA="hello")
        self.assertTrue("theta_GA param must be nonnegative float" in str(context.exception))

    def testParamThetaGAInv3(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_GA=-1.2)
        self.assertTrue("theta_GA param must be nonnegative float" in str(context.exception))

    def testParamThetaGA1(self):
        clf = eLCS(theta_GA=0)
        self.assertEqual(clf.theta_GA,0)

    def testParamThetaGA2(self):
        clf = eLCS(theta_GA=1)
        self.assertEqual(clf.theta_GA,1)

    def testParamThetaGA3(self):
        clf = eLCS(theta_GA=4.3)
        self.assertEqual(clf.theta_GA,4.3)


    def testParamThetaDelInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_del="hello")
        self.assertTrue("theta_del param must be nonnegative integer" in str(context.exception))

    def testParamThetaDelInv2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_del=2.3)
        self.assertTrue("theta_del param must be nonnegative integer" in str(context.exception))

    def testParamThetaDelInv3(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_del=-1.2)
        self.assertTrue("theta_del param must be nonnegative integer" in str(context.exception))

    def testParamThetaDelInv4(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_del=-5)
        self.assertTrue("theta_del param must be nonnegative integer" in str(context.exception))

    def testParamThetaDel1(self):
        clf = eLCS(theta_del=0)
        self.assertEqual(clf.theta_del,0)

    def testParamThetaDel2(self):
        clf = eLCS(theta_del=5)
        self.assertEqual(clf.theta_del,5)


    def testParamThetaSubInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_sub="hello")
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSubInv2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_sub=2.3)
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSubInv3(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_sub=-1.2)
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSubInv4(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(theta_sub=-5)
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSub1(self):
        clf = eLCS(theta_sub=0)
        self.assertEqual(clf.theta_sub,0)

    def testParamThetaSub2(self):
        clf = eLCS(theta_sub=5)
        self.assertEqual(clf.theta_sub,5)


    def testDoSubInvalid(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(doSubsumption=2)
        self.assertTrue("doSubsumption param must be boolean" in str(context.exception))

    def testDoSub(self):
        clf = eLCS(doSubsumption=True)
        self.assertEqual(clf.doSubsumption,True)


    def testSelectionInvalid(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(selectionMethod="hello")
        self.assertTrue("selectionMethod param must be 'tournament' or 'roulette'" in str(context.exception))

    def testSelection1(self):
        clf = eLCS(selectionMethod="tournament")
        self.assertEqual(clf.selectionMethod,"tournament")

    def testSelection2(self):
        clf = eLCS(selectionMethod="roulette")
        self.assertEqual(clf.selectionMethod,"roulette")


    def testRandomSeedInv1(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(randomSeed="hello")
        self.assertTrue("randomSeed param must be integer or 'none'" in str(context.exception))

    def testRandomSeedInv2(self):
        with self.assertRaises(Exception) as context:
            clf = eLCS(randomSeed=1.2)
        self.assertTrue("randomSeed param must be integer or 'none'" in str(context.exception))

    def testRandomSeed(self):
        clf = eLCS(randomSeed=-100)
        self.assertEqual(clf.randomSeed,-100)

    def testRandomSeed2(self):
        clf = eLCS(randomSeed=200)
        self.assertEqual(clf.randomSeed,200)

    def testRandomSeed3(self):
        clf = eLCS(randomSeed='none')
        self.assertEqual(clf.randomSeed,'none')

    '''SECTION 2: TEST eLCS Performance
        Testing for
        -Final balanced accuracy
        -Final time to train
        -Final macro & micro population sizes
        -Final Average Generality
        
        Across different # of iterations (1000,5000,10000)
    '''

    #Check X and Y must be numeric for fit method

    #Check Y must be discrete for fit method (eLCS works best only on classification problems)

    #Test balanced accuracy, times for binary attribute/phenotype training data (MP problems)

    #Test balanced accuracy, times for binary attribute/phenotype training data w/ missing data

    #Test balanced accuracy, times for continuous attribute training data

    #Test balanced accuracy, times for continuous attribute training data w/ missing data

    #Test balanced accuracy, times for binary attribute/phenotype testing data (MP problems w/ CV)

    #Test balanced accuracy, times for continuous attribute training data (w/ CV)
