from re import U
import torch
from math import pi
import matplotlib.pyplot as plt
import torch.linalg
import torch.distributions as distributions

def set_up_model(self,image_path):
# N,K,control_func,noiselevel,sol_func
	N = self.N
	Nt = self.Nt
	N_control = self.N_control
	Nt_control = self.Nt_control
	control_func = self.control_func
	noiselevel = self.noiselevel
	
	A = torch.zeros(N-1,N-1)
	k = 1/Nt
	h = 1/N
	mathcal_K = 0.5
	r = (mathcal_K*k)/(h**2)

	for i in range(N-1):
		if i == 0:
			A[i,i]=1+2*r
			A[i,i+1]=-r
		elif i == N-2:
			A[i,i-1]=-r
			A[i,i]=1+2*r
		else:
			A[i,i-1]=-r
			A[i,i]=1+2*r
			A[i,i+1]=-r

	G = torch.linalg.pinv(A)

	u_all = torch.zeros(N+1,Nt+1) #0- 1 - N-1 -N

	# initial condition 
	u_all[:,0] = torch.zeros(N+1)
	for i in range(N+1):
		if i>=0.2*N and i<=0.5*N:
			u_all[i,0] = 1
		else:
			u_all[i,0] = 0

	# generate temperature at all time and place 
	for i in range(1,Nt+1):
		u_all[1:N,i] = torch.matmul(G,u_all[1:N,i-1])

	# boundary values: u_all[0,:],u_all[N,:]
	u_all[0,:] = torch.zeros(Nt+1)
	u_all[N,:] = torch.zeros(Nt+1)

	# compute w_exact
	w_exact = []
	x = torch.linspace(0,1,N_control+1)
	t = torch.linspace(0,1,Nt_control+1)
	w_exact_dic = {'x': control_func['x'](x),'t': control_func['t'](t)}
	for i in w_exact_dic['x']:
		for j in w_exact_dic['t']:
			w_exact.append([i,j])
	w_exact = torch.FloatTensor(w_exact) # size: (N_control+1)(Nt_control+1) * 2
	
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
	fig1.savefig(image_path+'/w_exact.jpg')

	# compute temperature at control
	p = torch.zeros(N_control+1,Nt_control+1)
	observations = torch.zeros(N_control+1,Nt_control+1)

	for i in range(N_control+1):
		for j in range(Nt_control+1):
			a = torch.round(w_exact_dic['x'][i]*N).to(torch.int)
			b = torch.round(w_exact_dic['t'][j]*Nt).to(torch.int)
			p[i,j] = u_all[a,b]

	if noiselevel > 0:
		gamma = noiselevel*torch.eye(Nt_control)
		noise = distributions.MultivariateNormal(torch.zeros(Nt_control),gamma).sample()
	else:
		gamma = torch.eye(Nt_control+1)
		noise = torch.zeros(Nt_control+1)

	observations = p+noise

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
	fig2.savefig(image_path+'/observations.jpg')

	self.A = A
	self.G = G
	self.observations = observations
	self.w_exact = w_exact
	self.u_all = u_all
	self.p = p
	self.noise = noise
	self.gamma = gamma
	#A,G,observations,u_exact,p,noise,gamma
	return