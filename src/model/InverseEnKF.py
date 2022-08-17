from types import NoneType
import torch
from src.helper_funcs.moments import moments
from src.helper_funcs.early_stopping import early_stopping
import torch.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
from src.simulation.set_up_model import HeatModel2D
from src.helper_funcs.covmat import covmat

class InverseEnKFmodel():
	def __init__(self,heatmod,stopping,image_path):
		self.heatmod = heatmod
		self.E = []
		self.R = []
		self.M = []
		self.D = []
		self.theta_hat = []
		self.theta = torch.Tensor(np.array([heatmod.r,heatmod.r2])) # size 2 * Nt
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
		En = torch.zeros(1+2*self.heatmod.observations.size(dim=0),ensembleSize)
		En = torch.cat((self.heatmod.observations[:,0],self.heatmod.observations[:,0],torch.zeros(1)),dim=0)[:,None] + 20*torch.randn(En.shape)
		# En = torch.abs(En)

		self.En = En.type(torch.FloatTensor)
		self.m1,self.m2 = moments(self.En)

	def convergence(self,i):
		En = self.En
		theta = self.theta[:,i]

		l = self.heatmod.observations.size(dim=1)

		r2_hat = En[-1:,:]
		r1_hat = torch.mean(torch.div(En[:l,:],En[l:2*l,:]),dim=0)
		r1_hat = torch.mul(r1_hat,r2_hat)

		En = torch.cat((r1_hat,r2_hat),dim=0)
		m1 = torch.mean(En,dim=1)

		e = En - m1[:,None]
		r = En - theta[:,None]

		Ri = sum(pow(linalg.vector_norm(r,dim=0),2))\
		/ (linalg.vector_norm(theta)**2)
		(self.R).append(Ri/self.ensembleSize)

		Ei = sum(pow(linalg.vector_norm(e,dim=0),2))\
		/ (linalg.vector_norm(m1)**2)
		print(Ei/self.ensembleSize)
		(self.E).append(Ei/self.ensembleSize)

		Mi = linalg.vector_norm(m1-theta)
		(self.M).append(Mi)

		(self.D).append(m1-theta)
		print('descrepency:%f,%f'%((m1-theta)[0],(m1-theta)[1]))
		(self.theta_hat).append(m1)

		return

	def update_ensemble(self):

		print("Begin updating EnKF model...")

		A,B,C = self.A, self.B, self.C
		G = torch.cat((A,B,C[:,None]),dim=1).type(torch.FloatTensor)

		for i in range(self.heatmod.Nt):

			self.convergence(i)
			if i>0:
				if early_stopping(self.stopping,i,self.R[i],self.R[i-1],self.E[i]):
					break

			# Tk
			Cup = covmat(self.En,torch.mm(G,self.En)) # N * N
			Cpp = covmat(torch.mm(G,self.En),torch.mm(G,self.En))

			Tk = self.heatmod.observations[:,i]
			delta_Tk = self.heatmod.observations[:,i+1] - Tk

			for j in range(self.ensembleSize):
				try:
					X = linalg.pinv(Cpp + self.heatmod.gamma)
				except:
					print('early stopping at %d-th iter due to numerical error'%(i+1))
					break
				temp = torch.matmul(X,delta_Tk - torch.matmul(G,self.En[:,j]) )
				self.En[:,j] = self.En[:,j] + torch.matmul(Cup,temp)

			self.m1,self.m2 = moments(self.En)

			if i % 10 == 0:
				print('the %d-th iter done'%(i))

		self.convergence(i)
		self.final_plots(i)

	def final_plots(self,iter):
		ltype = '-x'
		color = 'b'
		image_path=self.image_path

		# write information to file
		# f = open(image_path+"/result.txt", "w")
		# f.write("New record appended:")
		# f.write('\n')
		# f.write("Stopping at %d-th iteration"%(iter))
		# f.write('\n')
		# f.write("Descrepency: %f, %f"%(self.D[-1].numpy()[0],self.D[-1].numpy()[1]))
		# f.write('\n')
		# f.write("E,R,M: %f, %f, %f"%(self.E[-1].numpy(),self.R[-1].numpy(),self.M[-1].numpy()))
		# f.write('\n')
		# if self.theta[0].numpy()>1e-3:
		# 	m0 = (self.D[-1].numpy()[0])/(self.theta[0].numpy())
		# else:
		# 	m0 = (self.D[-1].numpy()[0])/(1e-3)
		# if self.theta[1].numpy()>1e-3:
		# 	m1 = (self.D[-1].numpy()[1])/(self.theta[1].numpy())
		# else:
		# 	m1 = (self.D[-1].numpy()[1])/(1e-3)
		# f.write("Relative Error: %f, %f"%(m0,m1))
		# f.write('\n')
		# f.close()

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
		D = np.array([c.numpy() for c in self.D])

		f, ax1 = plt.subplots(2,1,figsize=(20,10))
		for idx,ax in enumerate(ax1):
			ax.set_yscale('symlog')
			ax.plot(torch.linspace(0,iter+1,iter+2),np.absolute(D[:,idx]),ltype,color=color)
			ax.set_xlabel('Iteration')
			ax.set_ylabel('D')
			ax.set_title('D over iteration')

		f.savefig(image_path+'/estimation/D.jpg',bbox_inches='tight')

		plt.figure()
		plt.plot(torch.linspace(0,iter+1,iter+2),M)
		plt.ylabel('M')
		plt.xlabel('Iteration')
		plt.title('M over iteration')
		plt.savefig(self.image_path+'/estimation/M.jpg',bbox_inches='tight')


		# plot reconstruction of theta
		f, ax1 = plt.subplots(2,1,figsize=(20,10))
		theta_hat = np.array([c.numpy() for c in self.theta_hat])
		for idx,ax in enumerate(ax1):
			ax.set_yscale('symlog')
			ax.plot(torch.linspace(0,iter+1,iter+2),theta_hat[:,idx],'k-')
			ax.plot(torch.linspace(0,iter+1,iter+2),self.theta[idx,:],'r-')
			ax.set_xlabel('Iteration')
			ax.set_ylabel('theta_hat[%d]'%(idx))
			ax.set_title('Reconstruction of theta')
		plt.savefig(image_path+'/estimation/ReconstructionOfTheta.jpg',bbox_inches='tight')

		return

	def temperature_predict(self):

		N,Nt = self.heatmod.N, self.heatmod.Nt
		theta_hat = np.array([c.numpy() for c in self.theta_hat])
		
		r,r2 = theta_hat[-1,0],theta_hat[-1,1]
		r = float(r)
		r2 = float(r2)

		pred_heatmod = HeatModel2D(N, Nt,0,r,r2)
		pred_heatmod.check_numerical()
		T0 = self.heatmod.observations[:,-1].numpy()
		pred_heatmod.compute_all_T(True,T0)
		pred_heatmod.generate_animation(self.image_path+'/estimation')
		
		T_predict = np.concatenate(pred_heatmod.T_all).reshape(Nt+1,N**2)
		T_predict = torch.from_numpy(T_predict)

		T_all = np.concatenate(self.heatmod.T_all).reshape(Nt+1,N**2)
		T_all = torch.from_numpy(T_all)

		r = T_predict - T_all
		R = pow(linalg.vector_norm(r,dim=1),2)
		plt.figure()
		plt.plot(torch.linspace(0,Nt,Nt+1),R.numpy())
		plt.ylabel('error')
		plt.xlabel('time')
		plt.title('Temperature prediction error')
		plt.savefig(self.image_path+'/estimation/T_predict_error.jpg',bbox_inches='tight')

		return
