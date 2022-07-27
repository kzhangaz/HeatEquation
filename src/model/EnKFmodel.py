import torch
from src.helper_funcs.moments import moments
from src.helper_funcs.early_stopping import early_stopping
import torch.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np

class EnKFmodel():
	def __init__(self,heatmod,stopping,image_path):
		self.heatmod = heatmod
		self.E = []
		self.R = []
		self.M = []
		self.theta_hat = []
		self.theta = torch.Tensor([heatmod.r,heatmod.r2])
		self.stopping = stopping
		self.image_path = image_path
		A = heatmod.A.getValues(range((heatmod.N)**2), range((heatmod.N)**2))
		self.A = torch.from_numpy(A)
		B = heatmod.B.getValues(range((heatmod.N)**2), range((heatmod.N)**2))
		self.B = torch.from_numpy(B)
		C = heatmod.C.array
		self.C = torch.from_numpy(C)

	def set_up_ensemble(self,ensembleSize):
		self.ensembleSize = ensembleSize
		En = torch.zeros(self.theta.size(dim=0),ensembleSize)
		En = self.theta[:,None] + 2000*torch.randn(En.shape)

		self.En = En
		self.m1,self.m2 = moments(self.En)

	def convergence(self):
		En = self.En
		m1 = self.m1
		theta = self.theta

		e = En - m1[:,None]
		r = En - theta[:,None]

		Ri = sum(pow(linalg.vector_norm(r,dim=0),2))\
		/ (linalg.vector_norm(theta)**2)
		(self.R).append(Ri/self.ensembleSize)

		Ei = sum(pow(linalg.vector_norm(e,dim=0),2))\
		/ (linalg.vector_norm(m1)**2)
		print(Ei/self.ensembleSize)
		(self.E).append(Ei/self.ensembleSize)

		(self.M).append(m1-theta)
		print('misfit:%f,%f'%((m1-theta)[0],(m1-theta)[1]))
		(self.theta_hat).append(m1)

		return

	def update_ensemble(self):

		print("Begin updating EnKF model...")

		for i in range(self.heatmod.Nt):

			self.convergence()
			if i>0:
				if early_stopping(self.stopping,i,self.R[i],self.R[i-1]):
					break

			# Tk
			A,B,C = self.A, self.B, self.C
			Tk = self.heatmod.observations[:,i]
			# delta_Tk
			delta_Tk = self.heatmod.observations[:,i+1] - Tk
			delta_Tk = delta_Tk.to(torch.float32)
			# Hk
			Hk = torch.stack((torch.matmul(A,Tk),torch.matmul(B,Tk)+C)).t().to(torch.float32)
			# Kalman gain
			Pk = self.m2
			temp1 = torch.matmul(Pk,Hk.t())
			temp2 = torch.matmul(Hk,temp1)+self.heatmod.gamma
			K = torch.matmul(temp1,torch.linalg.pinv(temp2))
			# update
			for j in range(self.ensembleSize):
				temp = delta_Tk-torch.matmul(Hk,self.En[:,j])
				self.En[:,j] = self.En[:,j] + torch.matmul(K,temp)

			self.m1,self.m2 = moments(self.En)

			if i % 10 == 0:
				print('the %d-th iter done'%(i+1))

		self.convergence()
		self.final_plots(i)

	def final_plots(self,iter):
		ltype = '-x'
		color = 'b'
		image_path=self.image_path

		# plot E_T & R_T
		f, (ax1,ax2) = plt.subplots(2,1,figsize=(20,10))

		E_T = torch.Tensor(self.E)
		R_T = torch.Tensor(self.R)
		ax1.set_yscale('symlog')
		ax1.plot(torch.linspace(0,iter+1,iter+2),E_T.numpy(),ltype,color=color)
		ax1.set_title('error&var of estimation of T over iteration')
		ax1.set_xlabel('Iteration')
		ax1.set_ylabel('E')

		ax2.set_yscale('symlog')
		ax2.plot(torch.linspace(0,iter+1,iter+2),R_T.numpy(),ltype,color=color)
		ax1.set_xlabel('Iteration')
		ax1.set_ylabel('R')

		f.savefig(image_path+'/estimation/E and R.jpg',bbox_inches='tight')

		# plot theta
		M = np.array([c.numpy() for c in self.M])

		f, ax1 = plt.subplots(2,1,figsize=(20,10))
		for idx,ax in enumerate(ax1):
			ax.set_yscale('symlog')
			ax.plot(torch.linspace(0,iter+1,iter+2),M[:,idx],ltype,color=color)
			ax.set_xlabel('Iteration')
			ax.set_ylabel('M')
			ax.set_title('M over iteration')

		f.savefig(image_path+'/estimation/M.jpg',bbox_inches='tight')

		# plot reconstruction of theta
		f, ax1 = plt.subplots(2,1,figsize=(20,10))
		theta_hat = np.array([c.numpy() for c in self.theta_hat])
		for idx,ax in enumerate(ax1):
			ax.set_yscale('symlog')
			ax.plot(torch.linspace(0,iter+1,iter+2),theta_hat[:,idx],'k-')
			ax.plot(torch.linspace(0,iter+1,iter+2),(self.theta[idx])*torch.ones(iter+2),'r-')
			ax.set_xlabel('Iteration')
			ax.set_ylabel('theta_hat[%d]'%(idx))
			ax.set_title('Reconstruction of theta')
		plt.savefig(image_path+'/estimation/ReconstructionOfTheta.jpg',bbox_inches='tight')

		return
