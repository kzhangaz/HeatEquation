from sympy import true
import torch
from src import moments
import torch.linalg as linalg

# def is_sorted(x):
# 	sorted,_ = diag(x).sort(descending=true)
# 	return (sorted==x)

def set_up_ensemble(self,ensembleSize,initEnsemble):
	#ensembleSize,initEnsemble,u_exact,N,K,A
	self.ensembleSize = ensembleSize
	self.initEnsemble = initEnsemble
	w_exact = self.w_exact
	N = self.N
	Nt = self.Nt
	N_control = self.N_control
	Nt_control = self.Nt_control
	A = self.A

	l = (N_control+1)*(Nt_control+1)
	En = torch.zeros(l,2,ensembleSize)

	if initEnsemble == 'random':

		for j in range(l):
			En[j,0,:] = torch.mean(w_exact[:,0]) * torch.ones(ensembleSize) + 0.5*torch.randn(ensembleSize)
			En[j,1,:] = torch.mean(w_exact[:,1]) * torch.ones(ensembleSize) + 0.5*torch.randn(ensembleSize)
		
		self.En = En
		self.m1,self.m2 = moments.moments(self.En)

		return

	# if initEnsemble == 'KL':
	# 	beta = 10
	# 	C0 = beta * linalg.pinv(A-eye(N,K))
		
	# 	D,V = linalg.eig(C0);

	# 	if not is_sorted(diag(D)):
	# 		D,I = diag(D).sort(descending=True)
	# 		V = index_select(V,1,I)
			
	# 	D = mm(D , diag(square(randn(K))) )
		
	# 	for j in range(ensembleSize):
	# 		En[:,j] = sqrt(D[j,j] + mean(u_exact)) * V[j,:]

	# 	self.En = En
	# 	self.m1,self.m2 = moments.moments(self.En)

	# 	return
	

	# if initEnsemble == 'brownian':

	# 	beta = 10
	# 	C0 = beta * linalg.pinv(A-eye(N,K))
	# 	x = mean(u_exact) * ones(N)
	# 	En = x.repeat(ensembleSize,1) + mm(randn(ensembleSize,N),linalg.cholesky(C0))
	# 	En = t(En)

	# 	self.En = En
	# 	self.m1,self.m2 = moments.moments(self.En)

		return

	