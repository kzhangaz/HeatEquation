from torch import matmul,cat,zeros,eye,t,matrix_rank,mean
from torch import linalg
from src import covmat
from src.moments import moments
from src.compute_Atheta import compute_Atheta

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

def update_EnKF(self,stopping,image_path):
	print("running update_EnKF...")

	for i in range(self.Nt+1):

		self.convergence(i)

		if i > 0:
			if early_stopping(stopping,i,self.M[i],self.M[i-1],self.noise):
				print('stopping early by '+stopping+' at %d-th iteration'%(i))
				break

		A_theta = compute_Atheta(self.m1[:3],self.N,self.Nt)
		u = self.u

		Tk_next = zeros(self.l,self.ensembleSize)
		# time update
		for j in range(self.ensembleSize):
			Tk_next[:,j] = matmul(A_theta,self.En[3:,j])+self.m1[2]*u[:,i]

		Xk_next = cat((self.En[:3,:],Tk_next),dim=0)
		Xhat,Pk_next = moments(Xk_next)

		# measurement update
		K1 = Pk_next[:,3:]
		K3 = Pk_next[3:,3:]+self.gamma
		
		if K3.isinf().any() or K3.isnan().any():
			raise ValueError("K3 here contains NaN or Inf")
		
		K2 = linalg.pinv(K3)
		# try low rank version of pinv

		K = matmul(K1,K2)

		for j in range(self.ensembleSize):
			self.En[:,j] = self.En[:,j] + matmul(K,self.y[:,i]-Xhat[3:])

		self.m1,self.m2 = moments(self.En)

		if i % 10 == 0:
			print('the %d-th iter of %d'%(i+1,self.Nt+1))
			print("Pk: rank is %d/2404, mean is %e, norm is %e"%(linalg.matrix_rank(Pk_next),mean(Pk_next),linalg.norm(Pk_next)))
			print("K: rank is %d/2404, det is %e, norm is %e"%(linalg.matrix_rank(K3),linalg.det(K3),linalg.norm(K3)))

	
	self.convergence(i)
	
	self.final_plot(i,image_path)
	
	return