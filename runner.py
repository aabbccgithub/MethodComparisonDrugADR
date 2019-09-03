from predictorWrapper import PredictorWrapper
from  models import MultiSVM, KNN, CCAModel,RFModel,RandomModel,NeuNModel,GBModel,RSCCAModel,MFModel,LogisticModel
import sys
def runSVM():
    wrapper = PredictorWrapper()
    import const
    PLIST = [i for i in xrange(1,2)]
    for p in PLIST:
        const.SVM_C = p
        model = MultiSVM()
        print wrapper.evalAModel(model)

def runRF():
    wrapper = PredictorWrapper()
    PLIST = [10*i for i in xrange(1,2)]
    import const

    for p in PLIST:
        const.RF = p
        model = RFModel()
        print wrapper.evalAModel(model)

def runGB():
    wrapper = PredictorWrapper()
    PLIST = [10*i for i in xrange(1,2)]
    import const

    for p in PLIST:
        const.RF = p
        model = GBModel()
        print wrapper.evalAModel(model)

def runKNN():
    wrapper = PredictorWrapper()
    KLIST = [10*i for i in xrange(1,2)]
    import const
    for k in KLIST:
        const.KNN = k
        model = KNN()
        print wrapper.evalAModel(model)

def runCCA():
    wrapper = PredictorWrapper()
    NCLIST = [10*i for i in xrange(1,2)]
    import const
    for c in NCLIST:
        const.CCA = c
        model = CCAModel()
        print wrapper.evalAModel(model)

def runSCCA():
    wrapper = PredictorWrapper()
    NCLIST = [10*i for i in xrange(1,2)]
    import const
    for c in NCLIST:
        const.CCA = c
        model = RSCCAModel()
        print wrapper.evalAModel(model)

def runRandom():
    wrapper = PredictorWrapper()
    model = RandomModel()
    print wrapper.evalAModel(model)

def runMF():
    wrapper = PredictorWrapper()
    KLIST = [10*i for i in xrange(1,2)]
    import const
    for k in KLIST:
        const.N_FEATURE = k
        model = MFModel()
        print wrapper.evalAModel(model)

def runNeu():
    wrapper = PredictorWrapper()
    import const
    PLIST = [10*i for i in xrange(1,2)]
    for p in PLIST:
        const.NeuN_H1 = p
        model = NeuNModel()
        print wrapper.evalAModel(model)

def runLR():

    wrapper = PredictorWrapper()
    import const
    PLIST = [1 * i for i in xrange(1, 2)]
    for p in PLIST:
        const.SVM_C = p
        model = LogisticModel()
        print wrapper.evalAModel(model)

if __name__ == "__main__":
    methodName = "SVM"
    import sys
    try:
        inp = sys.argv[1]
        if len(inp) > 1:
            methodName = inp
    except:
        pass
    if methodName == "KNN":
        runKNN()
    # elif methodName == "CCA":
    #     runCCA()
    elif methodName == "RF":
        runRF()
    elif methodName == "SVM":
        runSVM()
    elif methodName == "RD":
        runRandom()
    elif methodName == "NN":
        runNeu()
    elif methodName == "GB":
        runGB()
    # elif methodName == "SCCA":
    #     runSCCA()
    # elif methodName == "MF":
    #     runMF()
    elif methodName == "LR":
        runLR()
    else:
        print "Method named %s is unimplemented."%methodName
    #runSVM()
    #runRF()
    #runKNN()
    #runCCA()
    #runRandom()
    #runNeu()