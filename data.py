from  io import open
import numpy as np
import const
import random

import utils



class DataLoader2():
    def loadFold(self,ifold):
        inputTrainP = np.loadtxt("%s/inputTrainP_%s" % (const.DATA_ROOT_2,ifold))
        outputTrainP = np.loadtxt("%s/outputTrainP_%s" % (const.DATA_ROOT_2,ifold))
        inputTesto = np.loadtxt("%s/inputTestP_%s" % (const.DATA_ROOT_2,ifold))
        outputTest = np.loadtxt("%s/outputTestP_%s" % (const.DATA_ROOT_2,ifold))
        return inputTrainP,outputTrainP,inputTesto,outputTest


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


class GenECFPData():
    @staticmethod
    def __convertBinStringToArray(bs):
        sz = len(bs)
        ar = np.zeros(sz,dtype=int)
        for i in xrange(sz):
            if bs[i] == "1":
                ar[i] = 1
        return ar

    def loadECFPLiuData(self):
        ECFPFeatures = utils.load_obj(const.ECFP_FEATURE_PATH)
        fADR = open(const.ECFP_ADR_PATH)
        Chems = []
        ADRs = []
        while True:
            line = fADR.readline()
            if line == "":
                break
            line = line.strip()
            parts = line.split("|")
            chem = self.__convertBinStringToArray(parts[4])
            adr = self.__convertBinStringToArray(parts[3])

            Chems.append(chem)
            ADRs.append(adr)
        fADR.close()

        if len(ECFPFeatures) != len(Chems):
            print "Fatal error. Missmatched data"
            exit(-1)


        fin = open(const.ECFP_INFO)
        self.N_DRUGS = int(fin.readline().split(":")[-1].strip())
        self.N_FEATURE = int(fin.readline().split(":")[-1].strip())
        self.MAX_ATOMS = int(fin.readline().split(":")[-1].strip())

        self.ECFPFeatures = ECFPFeatures
        self.Chems = Chems
        self.ADRs = ADRs


    def getTrainTestPathByIFold(self,ifold):
        pTrainECFeature = "%s/%s_ec_%s" % (const.KFOLD_FOLDER_EC, const.TRAIN_PREFIX_EC, ifold)
        pTrainChemFeature = "%s/%s_chem_%s" % (const.KFOLD_FOLDER_EC, const.TRAIN_PREFIX_EC, ifold)
        pTrainADRs = "%s/%s_ADR_%s" % (const.KFOLD_FOLDER_EC, const.TRAIN_PREFIX_EC, ifold)

        pTestECFeature = "%s/%s_ec_%s" % (const.KFOLD_FOLDER_EC, const.TEST_PREFIX_EC, ifold)
        pTestChemFeature = "%s/%s_chem_%s" % (const.KFOLD_FOLDER_EC, const.TEST_PREFIX_EC, ifold)
        pTestADRs = "%s/%s_ADR_%s" % (const.KFOLD_FOLDER_EC, const.TEST_PREFIX_EC, ifold)

        return pTrainECFeature,pTrainChemFeature,pTrainADRs,pTestECFeature,pTestChemFeature,pTestADRs

    def exportKFold(self):
        foldSize = self.N_DRUGS / const.KFOLD
        ar = np.arange(0,self.N_DRUGS)
        random.seed(1)
        random.shuffle(ar)

        for i in xrange(const.KFOLD):
            #pTrainECFeature, pTrainChemFeature, pTrainADRs, pTestECFeature, pTestChemFeature, pTestADRs\
            paths = self.getTrainTestPathByIFold(i)

            arTrain = []
            arTest = []

            start = i * foldSize
            end = (i + 1) * foldSize
            if i == const.KFOLD - 1:
                end = self.N_DRUGS
            for jj in xrange(self.N_DRUGS):
                ar = arTrain
                if jj >= start and jj < end:
                    ar = arTest
                ix = ar[jj]
                ar.append([self.ECFPFeatures[ix],self.Chems[ix],self.ADRs[ix]])

            ars = [arTrain,arTest]

            for ii in xrange(2):
                for jj in xrange(3):
                    path = paths[ii*3+jj]
                    data = ars[ii][:][jj]
                    if jj == 0:
                        utils.save_obj(data,path)
                    else:
                        data = np.vstack(data)
                        np.savetxt(path,data)


















if __name__ == "__main__":
    #data = DataLoader()
    #data.exportKFold()

    data = GenECFPData()
    data.loadECFPLiuData()
    data.exportKFold()



