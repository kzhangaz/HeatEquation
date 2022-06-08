from torch import *
from math import pi
import matplotlib.pyplot as plt
import torch.linalg
import torch.distributions as distributions

def set_up_model(self,image_path):
# N,K,control_func,noiselevel,sol_func
	N = self.N
	K = self.K
	control_func = self.control_func
	noiselevel = self.noiselevel
	sol_func = self.sol_func
	
	L = zeros(N,K)
	L = ((K/pi)**2) * (2*eye(K) - \
		diag_embed(ones(K-1),1) - diag_embed(ones(K-1),-1))

	A = L + eye(N,K)
	G = torch.linalg.pinv(A)

	x = linspace(0,pi,N)
	u_exact = t(control_func(x))
	p = matmul(G,u_exact)

	if noiselevel > 0:
		gamma = noiselevel*eye(K)
		noise = distributions.MultivariateNormal(zeros(K),gamma).sample()
	else:
		gamma = eye(K)
		noise = zeros(K)

	observations = p+noise

	# plot
	fig=plt.figure()
	# ax=fig.add_axes([0,0,1,1])
	# ax.plot(x,observations,'r')
	# ax.legend(labels=('Exact solution','Noisy data'))
	ax1=fig.add_axes([0.1,0.1,0.8,0.8])
	ax1.plot(x,observations,'r')
	ax1.plot(x,sol_func(x),'k')
	fig.savefig(image_path+'/observations.jpg')

	self.A = A
	self.G = G
	self.observations = observations
	self.u_exact = u_exact
	self.p = p
	self.noise = noise
	self.gamma = gamma
	#A,G,observations,u_exact,p,noise,gamma
	return