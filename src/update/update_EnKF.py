from pickle import TRUE
from torch import mm,matmul
from torch import linalg
from src import covmat
from src import moments
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
				print('stopping early by '+stopping)
				break
		
		Cup = covmat.covmat(self.En,mm(self.G,self.En)) # N * N
		Cpp = covmat.covmat(mm(self.G,self.En),mm(self.G,self.En))

		for j in range(self.ensembleSize):
			temp = matmul(linalg.inv(Cpp + self.gamma),\
						self.observations - matmul(self.G,self.En[:,j]) )
			self.En[:,j] = self.En[:,j] + matmul(Cup,temp)

		self.m1,self.m2 = moments.moments(self.En)

		if ((i+1)/maxit * 100) % 10 == 0:
			print('the %d-th iter of %d'%(i+1,maxit))
	
	self.convergence()
	
	self.final_plot(i,image_path,method=1)
	
	return