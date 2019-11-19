class ClassAccuracy:
    def __init__(self):
        """ Initialize the accuracy calculation for a single class """
        self.T_myClass = 0  # For binary class problems this would include true positives
        self.T_otherClass = 0  # For binary class problems this would include true negatives
        self.F_myClass = 0  # For binary class problems this would include false positives
        self.F_otherClass = 0  # For binary class problems this would include false negatives

    def updateAccuracy(self, thisIsMe, accurateClass):
        """ Increment the appropriate cell of the confusion matrix """
        if thisIsMe and accurateClass:
            self.T_myClass += 1
        elif accurateClass:
            self.T_otherClass += 1
        elif thisIsMe:
            self.F_myClass += 1
        else:
            self.F_otherClass += 1

    def reportClassAccuracy(self):
        """ Print to standard out, summary on the class accuracy. """
        print("-----------------------------------------------")
        print("TP = " + str(self.T_myClass))
        print("TN = " + str(self.T_otherClass))
        print("FP = " + str(self.F_myClass))
        print("FN = " + str(self.F_otherClass))
        print("-----------------------------------------------")