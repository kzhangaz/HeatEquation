from torch import *
import torch.linalg as linalg
from src.predict_by_En import predict_by_En

def convergence(self):
	#En,u_exact,m1,G,p,noise,gamma
	En = self.En # l*2*ensemblesize
	w_exact = self.w_exact #l*2
	m1 = mean(En,dim=2) #l*2
	# G = self.G
	# p = self.p
	# noise = self.noise
	# gamma = self.gamma
	observations = self.observations # N_control+1 * Nt_control+1

	e = En - m1[:,:,None] # l*2*ensembleSize
	r = En - w_exact[:,:,None] # l*2*ensembleSize

	misfit = self.predict_by_En(En) - observations[:,:,None]
	# N_control+1 * Nt_control+1 *ensembleSize
	# misfit = mm(G,r) - noise[:,None]

	Eix = sum(pow(linalg.vector_norm(e[:,0,:],dim=0),2))\
		/ (linalg.vector_norm(m1[:,0])**2)
	Eix = Eix/self.ensembleSize
	Eit = sum(pow(linalg.vector_norm(e[:,1,:],dim=0),2))\
		/ (linalg.vector_norm(m1[:,1])**2)
	Eit = Eit/self.ensembleSize
	(self.E).append([Eix,Eit])


	Rix = sum(pow(linalg.vector_norm(r[:,0,:],dim=0),2))\
		/ (linalg.vector_norm(w_exact[:,0])**2)
	Rix = Rix/self.ensembleSize
	Rit = sum(pow(linalg.vector_norm(r[:,1,:],dim=0),2))\
		/ (linalg.vector_norm(w_exact[:,1])**2)
	Rit = Rit/self.ensembleSize
	(self.R).append([Rix,Rit])


	# ae = mm(mm(sqrt(linalg.pinv(gamma)),G),e)
	# AEi = sum(pow(linalg.vector_norm(ae,dim=0),2))\
	# 	/ (linalg.vector_norm(matmul(G,m1))**2)
	# (self.AE).append(AEi/self.ensembleSize)

	# ar = mm(mm(sqrt(linalg.pinv(gamma)),G),r)
	# ARi = sum(pow(linalg.vector_norm(ar,dim=0),2))\
	# 	/ (linalg.vector_norm(p)**2)
	# (self.AR).append(ARi/self.ensembleSize)

	misfit = flatten(misfit,start_dim=0,end_dim=1)
	Mi = sum(pow(linalg.vector_norm(misfit,dim=0),2))
	(self.M).append(Mi/self.ensembleSize)

	return