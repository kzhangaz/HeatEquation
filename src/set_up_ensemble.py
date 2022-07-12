import torch
from src import moments

# def is_sorted(x):
# 	sorted,_ = diag(x).sort(descending=true)
# 	return (sorted==x)

def set_up_ensemble(self,ensembleSize,initEnsemble):
	#ensembleSize,initEnsemble,u_exact,N,K,A
	self.ensembleSize = ensembleSize
	self.initEnsemble = initEnsemble
	T = self.T
	l = self.l
	theta = self.theta

	En = torch.zeros(l+3,ensembleSize)

	if initEnsemble == 'random':

		for j in range(3):
			En[j,:] = theta[j]*torch.ones(ensembleSize) + 15*torch.randn(ensembleSize)

		for j in range(3,l):
			En[j,:] = torch.mean(T[j,:]) * torch.ones(ensembleSize) + 15*torch.randn(ensembleSize)
			En[j,:] = torch.mean(T[j,:]) * torch.ones(ensembleSize) + 15*torch.randn(ensembleSize)
		
		self.En = En
		self.m1,self.m2 = moments.moments(self.En)
		# (self.theta_hat).append(self.m1[:3])

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

	