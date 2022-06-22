from torch import *
import torch.linalg as linalg
from src.compute_Atheta import compute_Atheta

def convergence(self,iter):

	En = self.En 
	m1 = self.m1
	noise = self.noise[:,iter]
	# gamma = self.gamma
	T = self.T[:,iter]
	theta = self.theta

	Atheta = compute_Atheta(theta,self.N,self.Nt)
	(self.theta_hat).append(m1[:3])

	e_theta = En[:3,:]-m1[:3,None]
	e_T = En[3:,:] - m1[3:,None] 
	r_theta = En[:3,:] - theta[:,None]
	r_T = En[3:,:] - T[:,None]

	misfit = mm(Atheta,r_T) - noise[:,None]

	E_theta = div(mean(e_theta**2,dim=1),m1[:3]**2)
	(self.E_theta).append(E_theta)

	E_T = sum(pow(linalg.vector_norm(e_T,dim=0),2))\
		/ (linalg.vector_norm(m1[3:])**2)
	E_T = E_T/self.ensembleSize
	(self.E_T).append(E_T)

	R_theta = div(mean(r_theta**2,dim=1),theta**2)
	(self.R_theta).append(R_theta)

	R_T = sum(pow(linalg.vector_norm(r_T,dim=0),2))\
		/ (linalg.vector_norm(T)**2)
	R_T = R_T/self.ensembleSize
	(self.R_T).append(R_T)

	# ae = mm(mm(sqrt(linalg.pinv(gamma)),G),e)
	# AEi = sum(pow(linalg.vector_norm(ae,dim=0),2))\
	# 	/ (linalg.vector_norm(matmul(G,m1))**2)
	# (self.AE).append(AEi/self.ensembleSize)

	# ar = mm(mm(sqrt(linalg.pinv(gamma)),G),r)
	# ARi = sum(pow(linalg.vector_norm(ar,dim=0),2))\
	# 	/ (linalg.vector_norm(p)**2)
	# (self.AR).append(ARi/self.ensembleSize)

	Mi = sum(pow(linalg.vector_norm(misfit,dim=0),2))
	(self.M).append(Mi/self.ensembleSize)

	return