import torch
def compute_Atheta(theta,N,Nt):

	mathcal_K,a,b = theta[0],theta[1],theta[2]
	h = 1/N
	ht = 1/Nt
	l = (N-1)**2

	r1 = (mathcal_K*ht)/(a * (h**2))
	r2 = (b*ht) / a

	A = torch.zeros(l,l)

	Aii = torch.zeros(N-1,N-1)
	Aii = Aii.fill_diagonal_((1-4*r1-r2).to(torch.float32))
	Aii = torch.diagonal_scatter(Aii,r1*torch.ones(N-2), 1)
	Aii = torch.diagonal_scatter(Aii,r1*torch.ones(N-2), -1)

	Aij1 = r1*torch.eye(N-1)
	Azero = torch.zeros(N-1,N-1)

	for i in range(N-1):
		if i==0:
			X2 = torch.tile(Azero, (1, N-3))
			A[0:N-1,:] = torch.cat((Aii,Aij1,X2),1)
		elif i==N-2:
			X1 = torch.tile(Azero, (1, N-3))
			A[i*(N-1):,:] = torch.cat((X1,Aii,Aij1),1)
		else:
			X1 = torch.tile(Azero, (1, i-1))
			X2 = torch.tile(Azero, (1, N-i-3))
			A[i*(N-1):(i+1)*(N-1),:] = torch.cat((X1,Aij1,Aii,Aij1,X2),1)
	
	return A