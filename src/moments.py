import torch
from src.vecmul import vecmul

def moments(En):

	#En size: l*2*Ensemblesize
	l = En.size(dim=0)
	m1 = torch.mean(En,dim=2) #l*2
	m2 = torch.zeros(l,l,2) #l*l*2
	
	for j in range(En.size(dim=2)):
		
		temp = vecmul(En[:,0,j],En[:,0,j])
		m2[:,:,0] = m2[:,:,0] + temp
		m2[:,:,0] = m2[:,:,0]/En.size(dim=2)
	
	for j in range(En.size(dim=2)):
		
		temp = vecmul(En[:,1,j],En[:,1,j])
		m2[:,:,1] = m2[:,:,1] + temp
		m2[:,:,1] = m2[:,:,1]/En.size(dim=2)
	
	return m1,m2

