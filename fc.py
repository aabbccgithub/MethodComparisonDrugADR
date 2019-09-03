import torch
import torch.nn as nn

if __name__ == "__main__":
    dimFeature = 20
    numRow= 8
    numInput = 10
    numFilter = 2
    CH_NUM_1 = 9
    numInput = 20
    outFeature = 12
    layer1 = torch.randn([ numFilter,dimFeature, CH_NUM_1],requires_grad=True)
    input = torch.randn([numInput,1,numRow,dimFeature],requires_grad=True)
    re = torch.matmul(input,layer1)
    lin = torch.randn(CH_NUM_1,outFeature,requires_grad=True)
    re2 = torch.matmul(re,lin)
    a = torch.zeros(1,requires_grad=True)
    re3 = torch.add(re2,a)
    print re3.shape

