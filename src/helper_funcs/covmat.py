import torch

def covmat(X,Y):

	ensembleSize = X.size(dim=1)
	N = X.size(dim=0)

	Xbar = torch.mean(X,dim=1) # N
	Ybar = torch.mean(Y,dim=1) # N

	X = (X - Xbar[:,None]).t()
	Y = (Y - Ybar[:,None]).t()

	C = torch.matmul(X[:,:,None],Y[:,None,:])
	C = torch.mean(C,dim=0)

	return C

