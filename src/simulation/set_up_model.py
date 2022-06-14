import torch
import matplotlib.pyplot as plt
import torch.linalg
import torch.distributions as distributions

def set_up_model(self,image_path):
	# N,K,control_func,noiselevel,sol_func

	N = self.N
	Nt = self.Nt
	control_func = self.control_func
	noiselevel = self.noiselevel
	
	ht = 1/Nt
	h = 1/N
	l = (N-1)**2
	A = torch.zeros(l,l)
	T = torch.zeros(l,Nt+1)
	y = torch.zeros(l,Nt+1)
	
	# parameters
	mathcal_K = 0.2
	a = 1
	b = 0.1
	theta = torch.tensor([mathcal_K,a,b])

	r1 = (mathcal_K*ht)/(a * (h**2))
	r2 = (b*ht) / a

	# compute A
	Aii = torch.zeros(N-1,N-1)
	Aii = Aii.fill_diagonal_(1-4*r1-r2)
	Aii = Aii.diagonal_scatter(r1*torch.ones(3), 1)
	Aii = Aii.diagonal_scatter(r1*torch.ones(3), -1)

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
	
	# initialize T0
	temp = torch.zeros(N-1)
	temp[int(0.2*N):int(0.5*N)]=torch.ones(int(0.5*N)-int(0.2*N))
	T[:,0] = torch.tile(temp,(1,N-1))
	
	# compute Tk
	for i in range(1,Nt+1):
		T[:,i] = torch.matmul(A,T[:,i-1])

	if noiselevel > 0:
		gamma = noiselevel*torch.eye(l)
		noise = distributions.MultivariateNormal(torch.zeros(l),gamma).sample_n(Nt+1)
		noise = torch.t(noise)

	else:
		gamma = torch.eye(l)
		noise = torch.zeros(l,Nt+1)

	y = T + noise




	# plot w_exact
	fig1 = plt.figure()
	for i in torch.linspace(0,1,Nt+1):
		plt.scatter(torch.linspace(0,1,N+1),i*torch.ones(N+1),s=5, c='k', marker='o')
	for i in w_exact_dic['t']:
		plt.scatter(w_exact_dic['x'],i*torch.ones(N_control+1),s=5, c='r', marker='o')
	plt.xlim((-0.02,1.02))
	plt.ylim((-0.02,1.02))
	plt.xlabel('x')
	plt.ylabel('time')
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(r'Discrete Grid $\Omega_h,$ h= %s, k=%s'%(h,k),fontsize=24,y=1.08)
	fig1.savefig(image_path+'/w_exact.jpg',bbox_inches='tight')

	# plot
	fig2 = plt.figure(figsize=(12,6))
	plt.subplot(121)
	for j in range(Nt_control+1):
	    plt.plot(w_exact_dic['x'],observations[:,j],'o:',label='t[%s]=%.2f'%(j,t[j].numpy()))

	plt.xlabel('x')
	plt.ylabel('w')
	plt.legend(bbox_to_anchor=(-.4, 1), loc=2, borderaxespad=0.)

	plt.subplot(122)
	plt.imshow(observations)
	plt.xticks(range(N_control+1), w_exact_dic['x'].numpy())
	plt.yticks(range(Nt_control+1), w_exact_dic['t'].numpy())
	plt.xlabel('x')
	plt.ylabel('time')
	clb=plt.colorbar()
	clb.set_label('observations at control (u)')
	plt.suptitle('Numerical Solution of the  Heat Equation r=%s'%(round(r,3)),fontsize=24,y=1.08)
	fig2.tight_layout()
	fig2.savefig(image_path+'/observations.jpg',bbox_inches='tight')

	self.A = A
	self.G = G
	self.observations = observations
	self.w_exact = w_exact
	self.w_exact_dic = w_exact_dic
	self.u_all = u_all
	self.p = p
	self.noise = noise
	self.gamma = gamma
	#A,G,observations,u_exact,p,noise,gamma
	return