import torch

def moments(En):

	#En size: l*Ensemblesize
	m1 = torch.mean(En,dim=1)
	l = En.size(dim=1)

	En = En-m1[:,None]
	En = En.t()

	m2 = torch.matmul(En[:,:,None],En[:,None,:])
	m2 = torch.mean(m2,dim=0)*(l/(l-1))
	
	return m1,m2

