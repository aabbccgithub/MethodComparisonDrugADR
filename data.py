from  io import open
import numpy as np
import const
import random
class DataLoader():
    def __init__(self):
        self.allChems = []
        self.allAdrs = []

        pass

    @staticmethod
    def __convertBinStringToArray(bs):
        sz = len(bs)
        ar = np.zeros(sz,dtype=int)
        for i in xrange(sz):
            if bs[i] == "1":
                ar[i] = 1
        return ar
    def loadLiuData(self):
        f = open(const.DATA_PATH)
        while True:
            line = f.readline()
            if line == "":
                break
            line = line.strip()
            parts = line.split("|")
            chem = self.__convertBinStringToArray(parts[4])
            adr = self.__convertBinStringToArray(parts[3])
            self.allAdrs.append(adr)
            self.allChems.append(chem)

        f.close()

        ndArrayChem = np.vstack(self.allChems)
        ndArrayADR = np.vstack(self.allAdrs)
        self.mergedData = np.concatenate([ndArrayChem,ndArrayADR],axis=1)

        self.nCHem = ndArrayChem.shape[0]
        self.nDInput =  ndArrayChem.shape[1]
        self.nDOutput = ndArrayADR.shape[1]
        if self.nDInput != const.INPUT_SIZE:
            print "Missmatch in dimensions"
            exit(-1)

    def getTrainTestPathByIFold(self,ifold):
        pTrain = "%s/%s_%s" % (const.KFOLD_FOLDER, const.TRAIN_PREFIX, ifold)

        pTest = "%s/%s_%s" % (const.KFOLD_FOLDER, const.TEST_PREFIX, ifold)
        return pTrain,pTest

    def exportKFold(self):
        self.loadLiuData()
        nChem = len(self.allChems)
        ar = np.ndarray(nChem,dtype=int)
        for i in xrange(nChem):
            ar[i] = i
        random.seed(1)
        random.shuffle(ar)
        self.mergedData = self.mergedData[ar]

        foldSize = nChem/const.KFOLD
        for i in xrange(const.KFOLD):
            pTrain, pTest = self.getTrainTestPathByIFold(i)
            arTrain = []
            arTest = []

            start = i * foldSize
            end = (i+1) * foldSize
            if i == const.KFOLD - 1:
                end = nChem
            for jj in xrange(nChem):
                ar = arTrain
                if jj >= start and jj < end:
                    ar = arTest
                ar.append(self.mergedData[jj])

            arTrain = np.vstack(arTrain)
            arTest = np.vstack(arTest)
            np.savetxt(pTrain,arTrain)
            np.savetxt(pTest,arTest)

    def splitMergeMatrix(self,mx):
        inputs,outputs = mx[:,:const.INPUT_SIZE], mx[:,const.INPUT_SIZE:]
        return inputs,outputs

    def loadFold(self,iFold):
        pTrain, pTest = self.getTrainTestPathByIFold(iFold)
        matTrain = np.loadtxt(pTrain)
        matTest = np.loadtxt(pTest)
        return matTrain, matTest





if __name__ == "__main__":
    data = DataLoader()
    data.exportKFold()



