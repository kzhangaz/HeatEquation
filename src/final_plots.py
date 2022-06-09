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
	# plot E
	f, (ax1,ax2) = plt.subplots(2,1)
	
	E = torch.Tensor(self.E)
	ax1.set_yscale('symlog')
	ax1.plot(torch.linspace(0,iter+1,iter+2),E[:,0].numpy(),ltype,color=color)
	ax1.set_title('E over iteration')
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('e(x)')

	ax2.set_yscale('symlog')
	ax2.plot(torch.linspace(0,iter+1,iter+2),E[:,1].numpy(),ltype,color=color)
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('e(t)')

	f.savefig(image_path+'/E.jpg',bbox_inches='tight')

	# plot R
	f, (ax1,ax2) = plt.subplots(2,1)
	R = torch.Tensor(self.R)
	ax1.set_yscale('symlog')
	ax1.plot(torch.linspace(0,iter+1,iter+2),R[:,0].numpy(),ltype,color=color)
	ax1.set_title('R over iteration')
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('r(x)')

	ax2.set_yscale('symlog')
	ax2.plot(torch.linspace(0,iter+1,iter+2),R[:,1].numpy(),ltype,color=color)
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('r(t)')

	f.savefig(image_path+'/R.jpg',bbox_inches='tight')

	# plot M
	plt.figure()
	plt.semilogy(torch.linspace(0,iter+1,iter+2),self.M,ltype,color=color)
	plt.xlabel('Iteration')
	plt.ylabel('var theta')
	plt.title('Misfit')
	plt.savefig(image_path+'/Misfit.jpg',bbox_inches='tight')

	# plot
	plt.figure()
	w_exact = self.w_exact 
	m1 = self.m1
	plt.scatter(w_exact[0:100,0],w_exact[0:100,1], s=5, c='b', marker='o')
	plt.scatter(m1[0:100,0],m1[0:100,1], s=5, c='r', marker='o')
	plt.title('Reconstruction of the control')
	plt.xlabel('x')
	plt.ylabel('t')
	plt.savefig(image_path+'/ReconstructionOfControl.jpg',bbox_inches='tight')

	return