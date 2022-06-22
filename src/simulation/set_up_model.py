import torch
import matplotlib.pyplot as plt
import torch.linalg
import torch.distributions as distributions
from src.simulation.set_up_control import set_up_control
from matplotlib.animation import FuncAnimation

def set_up_model(self,image_path):
	# N,K,control_func,noiselevel,sol_func

	N = self.N
	Nt = self.Nt
	noiselevel = self.noiselevel
	
	ht = 1/Nt #0.1
	h = 1/N #0.02
	l = (N-1)**2
	self.l = l
	A = torch.zeros(l,l)
	T = torch.zeros(l,Nt+1)
	
	# parameters
	mathcal_K = 0.02
	a = 25
	b = 20
	theta = torch.tensor([mathcal_K,a,b])
	self.theta = theta

	r1 = (mathcal_K*ht)/(a * (h**2))
	r2 = (b*ht) / a
	print("r1, r2 is %f, %f"%(r1,r2))

	if 4*r1+r2 >= 1:
		raise ValueError("4r1+r2 should be less than 1 to be numerically stable")

	# compute A
	Aii = torch.zeros(N-1,N-1)
	Aii = Aii.fill_diagonal_(1-4*r1-r2)
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
	
	# initialize T0
	temp = torch.zeros(N-1)
	temp[int(0.2*N):int(0.5*N)]=torch.ones(int(0.5*N)-int(0.2*N))
	a,b = int(0.2*N),int(0.5*N)
	X1 =  torch.tile(torch.zeros(N-1),(1,a))
	X2 = torch.tile(temp,(1,b-a))
	X3 = torch.tile(torch.zeros(N-1),(1,N-1-b))
	T[:,0] = torch.cat((X1,X2,X3),1)
	# draw initial state
	fig = plt.figure()
	plt.imshow(T[:,0].reshape(N-1,N-1))
	plt.xticks(range(N-1), [i/N for i in range(1,N)])
	plt.yticks(range(N-1), [i/N for i in range(1,N)])
	plt.xlabel('x')
	plt.ylabel('y')
	clb=plt.colorbar()
	clb.set_label('temperature')
	plt.title('Initial T0')
	fig.tight_layout()
	fig.savefig(image_path+'/simulation/Initial State.jpg',bbox_inches='tight')


	# set up control u
	test = 1 # 0: zero control 1: sth not zero
	u = set_up_control(Nt,N,l,test,image_path)
	self.u = u
	
	# compute Tk
	for i in range(1,Nt+1):
		T[:,i] = torch.matmul(A,T[:,i-1])+r2*u[:,i-1]
		# if i == int(Nt/2):
		# 	print("after control:")
		# if i%5:
		# 	print(torch.norm(T[:,i]))

	# plot T without noise
	def animate(k):
		# k from 0 to Nt
		plt.clf()

		plt.title("Temperature at t = %d unit time"%(k))
		plt.xlabel("x")
		plt.ylabel("y")

		C = T[:,k].reshape(N-1,N-1)

		# This is to plot u_k (u at time-step k)
		plt.pcolormesh(C, cmap=plt.cm.jet, vmin=0,vmax=3)
		plt.colorbar()
		if k%5 == 0:
			plt.savefig(image_path+'/simulation/T=%d'%(k))
		return plt

	anim = FuncAnimation(plt.figure(), animate, interval=200, frames=Nt+1, repeat=False)
	anim.save(image_path+'/simulation/heat_equation_simulation.gif', writer='imagemagick', fps=60)

	if noiselevel > 0:
		gamma = noiselevel*torch.eye(l)
		noise = distributions.MultivariateNormal(torch.zeros(l),gamma).sample_n(Nt+1)
		noise = torch.t(noise)

	else:
		gamma = torch.eye(l)
		noise = torch.zeros(l,Nt+1)
	
	y = T + noise


	self.A = A
	self.T = T
	self.y = y
	self.noise = noise
	self.gamma = gamma
	#A,G,observations,u_exact,p,noise,gamma
	return