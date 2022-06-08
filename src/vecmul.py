from torch import zeros

def vecmul(x,y):
	if x.dim()>1 or y.dim()>1:
		raise ValueError("input must be 1-dim vectors")

	if x.size() != y.size():
		raise ValueError("input vectors must be of same size!")
	
	N = x.size(dim=0)
	mat = zeros(N,N)
	for i,entry in enumerate(x):
			mat[i,:] = entry * y

	return mat
