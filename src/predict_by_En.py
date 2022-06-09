import torch

def predict_by_En(self,En):

	# En: l*2*ensemblesize
	# want y: (N_control+1)*(Nt_control+1) * ensemblesize
	u_all = self.u_all # (N+1)*(Nt+1)
	N = self.N
	Nt = self.Nt
	N_control = self.N_control
	Nt_control = self.Nt_control

	ensembleSize = En.size(dim=2)

	y = torch.zeros(N_control+1,Nt_control+1,ensembleSize)

	for en_idx in range(ensembleSize):
		# compute y[:,:,en_idx]
		thisEn = En[:,:,en_idx]
		
		for i in range(N_control+1):
			for j in range(Nt_control+1):
				idx = (Nt_control+1)*i+j
				x,t = thisEn[idx,0],thisEn[idx,1]
				a = torch.round(x*N).to(torch.int)
				b = torch.round(t*Nt).to(torch.int)
				if a<0:
					a=0
				if a>N:
					a=N
				if b<0:
					b=0
				if b>Nt:
					b=Nt
				y[i,j,en_idx] = u_all[a,b]

	return y
	
	# else:
		
	# 	thisEn = En
	# 	y = torch.zeros(N_control+1,Nt_control+1)
			
	# 	for i in range(N_control+1):
	# 		for j in range(Nt_control+1):
	# 			idx = (Nt_control+1)*i+j
	# 			x,t = thisEn[idx,0],thisEn[idx,1]
	# 			a = torch.round(x*N).to(torch.int)
	# 			b = torch.round(t*Nt).to(torch.int)
	# 			y[i,j] = u_all[a,b]

	# 	return y