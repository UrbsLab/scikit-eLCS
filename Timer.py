# Import Required Modules---------------
import time


# --------------------------------------

class Timer:
    def __init__(self):
        # Global Time objects
        self.globalStartRef = time.time()
        self.globalTime = 0.0
        self.addedTime = 0.0

        # Match Time Variables
        self.startRefMatching = 0.0
        self.globalMatching = 0.0

        # Deletion Time Variables
        self.startRefDeletion = 0.0
        self.globalDeletion = 0.0

        # Subsumption Time Variables
        self.startRefSubsumption = 0.0
        self.globalSubsumption = 0.0

        # Selection Time Variables
        self.startRefSelection = 0.0
        self.globalSelection = 0.0

        # Evaluation Time Variables
        self.startRefEvaluation = 0.0
        self.globalEvaluation = 0.0

        # ************************************************************

    def startTimeMatching(self):
        """ Tracks MatchSet Time """
        self.startRefMatching = time.time()

    def stopTimeMatching(self):
        """ Tracks MatchSet Time """
        diff = time.time() - self.startRefMatching
        self.globalMatching += diff

        # ************************************************************

    def startTimeDeletion(self):
        """ Tracks Deletion Time """
        self.startRefDeletion = time.time()

    def stopTimeDeletion(self):
        """ Tracks Deletion Time """
        diff = time.time() - self.startRefDeletion
        self.globalDeletion += diff

    # ************************************************************
    def startTimeSubsumption(self):
        """Tracks Subsumption Time """
        self.startRefSubsumption = time.time()

    def stopTimeSubsumption(self):
        """Tracks Subsumption Time """
        diff = time.time() - self.startRefSubsumption
        self.globalSubsumption += diff

        # ************************************************************

    def startTimeSelection(self):
        """ Tracks Selection Time """
        self.startRefSelection = time.time()

    def stopTimeSelection(self):
        """ Tracks Selection Time """
        diff = time.time() - self.startRefSelection
        self.globalSelection += diff

    # ************************************************************
    def startTimeEvaluation(self):
        """ Tracks Evaluation Time """
        self.startRefEvaluation = time.time()

    def stopTimeEvaluation(self):
        """ Tracks Evaluation Time """
        diff = time.time() - self.startRefEvaluation
        self.globalEvaluation += diff

        # ************************************************************

    def returnGlobalTimer(self):
        """ Set the global end timer, call at very end of algorithm. """
        self.globalTime = (time.time() - self.globalStartRef) + self.addedTime  # Reports time in minutes, addedTime is for population reboot.
        return self.globalTime

    def reportTimes(self):
        self.globalTime = (time.time() - self.globalStartRef) + self.addedTime
        outputTime = {"Global Time":str(self.globalTime),
                     "Matching Time":str(self.globalMatching),
                     "Deletion Time":str(self.globalDeletion),
                     "Subsumption Time":str(self.globalSubsumption),
                     "Selection Time":str(self.globalSelection),
                     "Evaluation Time":str(self.globalEvaluation)}

        return outputTime