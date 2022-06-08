from torch import mean
from torch import zeros
from src.vecmul import vecmul

def covmat(X,Y):
	ensembleSize = X.size(dim=1)
	N = X.size(dim=0)

	Xbar = mean(X,dim=1) # N
	Ybar = mean(Y,dim=1) # N

	C = zeros(N,N)
	for j in range(ensembleSize):
		# temp = zeros(N,N)
		# for i,entry in enumerate(X[:,j]-Xbar):
		# 	temp[i,:] = entry * (Y[:,j]-Ybar)
		temp = vecmul(X[:,j]-Xbar,Y[:,j]-Ybar)
		C = C + temp
		# C = C + mm(X[:,j]-Xbar,t(Y[:,j]-Ybar))

	C = C/ensembleSize
	return C
