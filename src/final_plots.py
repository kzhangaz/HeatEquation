import matplotlib.pyplot as plt
import torch
from math import pi

def final_plot(self,iter,image_path,method):
	if method == 1:
		ltype = '-x'
		color = 'b'
	elif method == 5:
		ltype = '-v'
		color = 'r'
	else:
		print("No such method")
		return
	
	# plot E, AE
	f, (ax1,ax2) = plt.subplots(2,1)
	
	ax1.set_yscale('symlog')
	ax1.plot(torch.linspace(0,iter+1,iter+2),self.E,ltype,color=color)
	ax1.set_title('Deviation from the mean')
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('v')
	plt.savefig(image_path+'/E.jpg')

	ax2.set_yscale('symlog')
	ax2.plot(torch.linspace(0,iter+1,iter+2),self.AE,ltype,color=color)
	ax2.set_xlabel('Iteration')
	ax2.set_ylabel('V')
	plt.savefig(image_path+'/AE.jpg')

	# plot R, AR
	f, (ax1,ax2) = plt.subplots(2,1)
	
	ax1.set_yscale('symlog')
	ax1.plot(torch.linspace(0,iter+1,iter+2),self.R,ltype,color=color)
	ax1.set_title('Residuals')
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('r')
	plt.savefig(image_path+'/R.jpg')

	ax2.set_yscale('symlog')
	ax2.plot(torch.linspace(0,iter+1,iter+2),self.AR,ltype,color=color)
	ax2.set_xlabel('Iteration')
	ax2.set_ylabel('R')
	plt.savefig(image_path+'/AR.jpg')

	# plot M
	plt.figure()
	plt.semilogy(torch.linspace(0,iter+1,iter+2),self.M,ltype,'Color')
	plt.semilogy([0,iter+1],[torch.linalg.vector_norm(self.noise)**2,torch.linalg.vector_norm(self.noise)**2],'k:')
	plt.xlabel('Iteration')
	plt.ylabel('var theta')
	plt.title('Misfit')
	plt.savefig(image_path+'/Misfit.jpg')

	# plot 
	x = torch.linspace(0,pi,self.N)
	plt.figure()
	# if method == 1:
	# 	plt.plot(x,self.sol_func(x),'k-')
	plt.plot(x,torch.matmul(self.G,self.m1),ltype,color=color)
	plt.title('Exact solution with u(x)=1 vs Reconstruction')
	plt.xlabel('x')
	plt.savefig(image_path+'/Reconstruction.jpg')

	# plot
	plt.figure()
	# if method == 1:
	# 	plt.plot(x,self.control_func(x),'k-')
	plt.plot(x,self.m1,ltype,color=color)
	plt.title('Reconstruction of the control')
	plt.xlabel('x')
	plt.savefig(image_path+'/ReconstructionOfControl.jpg')

	return