from torch import *
import torch.linalg as linalg

def convergence(self):
	#En,u_exact,m1,G,p,noise,gamma
	En = self.En
	u_exact = self.u_exact
	m1 =  mean(En,dim=1)
	G = self.G
	p = self.p
	noise = self.noise
	gamma = self.gamma

	e = En - m1[:,None]
	r = En - u_exact[:,None] # N * ensembleSize

	misfit = mm(G,r) - noise[:,None] # N * ensembleSize

	Ei = sum(pow(linalg.vector_norm(e,dim=0),2))\
		/ (linalg.vector_norm(m1)**2)
	(self.E).append(Ei/self.ensembleSize)

	Ri = sum(pow(linalg.vector_norm(r,dim=0),2))\
		/ (linalg.vector_norm(u_exact)**2)
	(self.R).append(Ri/self.ensembleSize)

	ae = mm(mm(sqrt(linalg.pinv(gamma)),G),e)
	AEi = sum(pow(linalg.vector_norm(ae,dim=0),2))\
		/ (linalg.vector_norm(matmul(G,m1))**2)
	(self.AE).append(AEi/self.ensembleSize)

	ar = mm(mm(sqrt(linalg.pinv(gamma)),G),r)
	ARi = sum(pow(linalg.vector_norm(ar,dim=0),2))\
		/ (linalg.vector_norm(p)**2)
	(self.AR).append(ARi/self.ensembleSize)

	Mi = sum(pow(linalg.vector_norm(misfit,dim=0),2))
	(self.M).append(Mi/self.ensembleSize)

	return