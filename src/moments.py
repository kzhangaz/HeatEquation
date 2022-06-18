import torch
from src.vecmul import vecmul

def moments(En):

	#En size: l*Ensemblesize
	m1 = torch.mean(En,dim=1)

	En = En-m1[:,None]
	En = En.t()

	m2 = torch.matmul(En[:,:,None],En[:,None,:])
	m2 = torch.mean(m2,dim=0)
	
	# for j in range(En.size(dim=1)):
		
	# 	temp = torch.matmul(En[:,None,j],En[None,:,j])
	# 	m2[:,:,0] = m2[:,:,0] + temp
	# 	m2[:,:,0] = m2[:,:,0]/En.size(dim=1)
	
	return m1,m2

