import matplotlib.pyplot as plt
import torch
from math import pi

def final_plot(self,iter,image_path,method):
	if method == 1:
		image_path = image_path+'/method1'
	elif method == 5:
		image_path = image_path+'/method5'
	else:
		print("No such method")
		return

	ltype = '-x'
	color = 'b'

	# plot E_T & R_T
	f, (ax1,ax2) = plt.subplots(2,1)
	
	E_T = torch.Tensor(self.E_T)
	R_T = torch.Tensor(self.R_T)
	ax1.set_yscale('symlog')
	ax1.plot(torch.linspace(0,iter+1,iter+2),E_T.numpy(),ltype,color=color)
	ax1.set_title('E_T over iteration')
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('E_T')

	ax2.set_yscale('symlog')
	ax2.plot(torch.linspace(0,iter+1,iter+2),R_T.numpy(),ltype,color=color)
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('R_T')

	f.savefig(image_path+'/estimation/estimation of T.jpg',bbox_inches='tight')

	# plot theta
	E_theta = torch.Tensor(self.E_theta)
	R_theta = torch.Tensor(self.R_theta)

	f, ax1 = plt.subplots(1,3)
	for idx,ax in enumerate(ax1):
		ax.set_yscale('symlog')
		ax.plot(torch.linspace(0,iter+1,iter+2),E_theta[:,idx].numpy(),ltype,color=color)
		ax.set_xlabel('Iteration')
		ax.set_ylabel('E_theta')
	ax.set_title('E_theta over iteration')

	f.savefig(image_path+'/estimation/E_theta.jpg',bbox_inches='tight')

	f, ax1 = plt.subplots(1,3)
	for idx,ax in enumerate(ax1):
		ax.set_yscale('symlog')
		ax.plot(torch.linspace(0,iter+1,iter+2),R_theta[:,idx].numpy(),ltype,color=color)
		ax.set_xlabel('Iteration')
		ax.set_ylabel('R_theta')
	ax.set_title('R_theta over iteration')

	f.savefig(image_path+'/estimation/R_theta.jpg',bbox_inches='tight')

	# plot M
	plt.figure()
	plt.semilogy(torch.linspace(0,iter+1,iter+2),self.M,ltype,color=color)
	plt.xlabel('Iteration')
	plt.ylabel('var theta')
	plt.title('Misfit')
	plt.savefig(image_path+'/Misfit.jpg',bbox_inches='tight')

	# plot reconstruction of theta
	f, ax1 = plt.subplots(1,3)
	theta_hat = self.theta_hat
	for idx,ax in enumerate(ax1):
		ax.set_yscale('symlog')
		ax.plot(torch.linspace(0,iter+1,iter+2),theta_hat[:,idx].numpy(),'k-')
		ax.plot(torch.linspace(0,iter+1,iter+2),(self.theta[idx])*torch.ones(iter+2),'r-')
		ax.set_xlabel('Iteration')
		ax.set_ylabel('theta_hat[%d]'%(idx))
	ax.set_title('Reconstruction of theta')
	plt.savefig(image_path+'/ReconstructionOfTheta.jpg',bbox_inches='tight')

	return