from torch import zeros,matmul

def vecmul(x,y):
	if x.dim()>1 or y.dim()>1:
		raise ValueError("input must be 1-dim vectors")

	if x.size() != y.size():
		raise ValueError("input vectors must be of same size!")
	
	# N = x.size(dim=0)
	# mat = zeros(N,N)

	x = x[:,None]
	y = y[None,:]
	mat = matmul(x,y)

	# for i,entry in enumerate(x):
	# 		mat[i,:] = entry * y

	return mat
