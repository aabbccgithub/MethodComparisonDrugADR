
library(matlib)

library(MASS)

trainPath = "/home/anhnd/DTI Project/Data/LiuData/kfolds/train__0"
testPath = "/home/anhnd/DTI Project/Data/LiuData/kfolds/test__0"

train = as.matrix(utils::read.delim(trainPath,sep=" "))
trainInput = train[,1:881]
trainOutput = train[,882:ncol(train)]

test = as.matrix(utils::read.delim(testPath,sep=" "))
testInput = test[,1:881]
testOutput = test[,882:ncol(train)]
library(PMA)
x <- CCA(trainInput,trainOutput,standardize=FALSE,K=20)
print (x)
print (dim(x$u))
print (dim(x$v))
print (dim(trainInput))

write.matrix(x$u,"/home/anhnd/DTI Project/Data/LiuData/kfolds/u_0")
write.matrix(x$v,"/home/anhnd/DTI Project/Data/LiuData/kfolds/v_0")

