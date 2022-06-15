import torch
import matplotlib.pyplot as plt

def set_up_control(Nt,N,l,test,image_path):
	
	if test == 0: 
		u = torch.zeros(l,Nt+1)
		return u
	
	if test == 1:
		u = torch.zeros(l,Nt+1)
		temp = torch.zeros(N-1)
		a = int(0.75*N)-int(0.7*N)
		temp[int(0.7*N):int(0.75*N)]= 300*torch.ones(a)
		u[:,int(Nt/2)] = torch.tile(temp,(1,N-1))

		fig = plt.figure()
		plt.imshow(u[:,int(Nt/2)].reshape(N-1,N-1))
		plt.xticks(range(N-1), [i/N for i in range(1,N)])
		plt.yticks(range(N-1), [i/N for i in range(1,N)])
		plt.xlabel('x')
		plt.ylabel('y')
		clb=plt.colorbar()
		clb.set_label('temperature')
		plt.title('temperature of control at time Nt/2')
		fig.tight_layout()
		fig.savefig(image_path+'/simulation/control.jpg',bbox_inches='tight')

		return u