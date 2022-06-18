from pickle import TRUE
from torch import mm,matmul,flatten
from torch import linalg
from src import covmat
from src import moments
from src.predict_by_En import predict_by_En

def early_stopping(stopping,i,Mi,Mi1,noise):

	#Discrepancy principle
	if stopping == 'discrepancy':
		if Mi <= (linalg.vector_norm(noise)**2):
			return True
		else:
			return False
	#Relative error
	if stopping == 'relative':
		tol = 1e-3
		if i > 1:
			if abs(Mi-Mi1) < tol:
				return True
			else:
				return False
		else:
			return False

def update_EnKF(self,maxit,stopping,image_path):
	print("running update_EnKF...")

	for i in range(int(maxit)):

		self.convergence()

		if i > 0:
			if early_stopping(stopping,i,self.M[i],self.M[i-1],self.noise):
				print('stopping early by '+stopping+' at %d-th iteration'%(i))
				break

		# En: l*2*ensembleSize, prediction: N_control+1 * Nt_control+1 * ensemblesize
		# Cup = covmat.covmat(self.En,mm(self.G,self.En)) # N * N

		# update for x,t
		Gu = flatten(self.predict_by_En(self.En),start_dim=0,end_dim=1)
		Cup_x = covmat.covmat(self.En[:,0,:],Gu)
		Cup_t = covmat.covmat(self.En[:,1,:],Gu)
		Cpp = covmat.covmat(Gu,Gu) # l*l

		for j in range(self.ensembleSize):
			temp = matmul(linalg.inv(Cpp + self.gamma),\
						flatten(self.observations,start_dim=0,end_dim=1)\
						 - Gu[:,j] )
			self.En[:,0,j] = self.En[:,0,j] + matmul(Cup_x,temp)
			self.En[:,1,j] = self.En[:,1,j] + matmul(Cup_t,temp)

		self.m1,self.m2 = moments.moments(self.En)

		if ((i+1)/maxit * 100) % 10 == 0:
			print('the %d-th iter of %d'%(i+1,maxit))
	
	self.convergence()
	
	self.final_plot(i,image_path,method=1)
	
	return