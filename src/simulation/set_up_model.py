from __future__ import division
import sys, petsc4py
import numpy as np
import matplotlib.pyplot as plt
import torch

petsc4py.init(sys.argv)

from petsc4py import PETSc
from matplotlib.animation import FuncAnimation

class HeatModel2D:

	def __init__(self, N, Nt, mode,r,r2):

		if mode!=0 and mode!=1:
			raise ValueError('mode should choose between 0 and 1!')
		self.mode = mode
		self.N = N
		self.Nt = Nt
		h = 1/(N-1)
		ht = 1/Nt
		self.r = r
		self.r2 = r2
		self.T_all = []

	def check_numerical(self):

		if self.mode == 0:
			if 1-4*(self.r) <= 0 or 1-4*(self.r)>=1:
				raise ValueError("Numerically Not Stable!")
			if 1-3*(self.r)+(self.r2) <= 0 or 1-3*(self.r)+(self.r2)>=1:
				raise ValueError("Numerically Not Stable!")
		elif self.mode == 1:
			if min(1-4*(self.r)) <= 0 or max(1-4*(self.r))>=1:
				raise ValueError("Numerically Not Stable!")
			if min(1-3*(self.r)+(self.r2)) <= 0 or max(1-3*(self.r)+(self.r2))>=1:
				raise ValueError("Numerically Not Stable!")
		
		print("Numerically OK! ;) Preparing for Computation")
		self.set_matA()
		self.set_matB()
		self.set_vecC()


	def set_matA(self):

		N = self.N
		l = (N)**2
		A = PETSc.Mat().createAIJ([l, l])
		A.setUp()

		diag = PETSc.Vec().createSeq(l)
		diag.setArray((-4)*np.ones(l))
		A.setDiagonal(diag)

		for i in range(1,N-1):
			A.setValue(i*N,i*N,-3)

		for i in range(1,N-1):
			A.setValue(i,i,-3)
			A.setValue(l-i-1,l-i-1,-3)

		A.setValue(0,0,-2)
		A.setValue((N-1)*N,(N-1)*N,-2)

		for i in range(l):
			if i>0: 
				if i%(N) != 0:
					A.setValue(i, i-1, 1)
				if i-(N) >= 0:
					A.setValue(i, i-N, 1)
			if i<(l-1):
				if i%(N) != N-1:
					A.setValue(i, i+1, 1)
				if i+N <= l-1:
					A.setValue(i, i+N, 1)

		for i in range(N):
			A.setValues((i+1)*N-1,np.arange(l,dtype=np.int32),np.zeros(l))

		A.assemble()
		self.A = A

		return


	def set_matB(self):

		N = self.N
		l = (N)**2
		B = PETSc.Mat().createAIJ([l, l])
		B.setUp()

		for i in range(N):
			B.setValue(i*N,i*N,1)

		B.assemble()
		self.B = B

		return

	def set_vecC(self):

		N = self.N
		l = (N)**2
		C = PETSc.Vec().createSeq(l)

		for i in range(N):
			C.setValue(i*N,1)

		self.C = C

		return

	def compute_delta_T(self, T,k):

		# Create a new sparse PETSc matrix, fill it and then assemble it
		# l = (self.N)**2
		r = self.r
		r2 = self.r2
		A = self.A
		B = self.B
		C = self.C

		if self.mode == 1:
			r,r2 = r[k],r2[k]
			r,r2 = float(r),float(r2)

		temp = r*A + r2*B
		delta_T = temp*T + r2*C

		return delta_T


	def compute_all_T(self,customize,T0):

		#initialize
		N = self.N
		l = N**2
		if customize == False:
			T = PETSc.Vec().createSeq(l)
			T.setArray(np.ones(l))
		else:
			T = PETSc.Vec().createSeq(l)
			T.setArray(T0)

		for i in range(N):
			T.setValue((i+1)*N-1,4)

		(self.T_all).append(T.array)

		for i in range(self.Nt):
			T_delta = self.compute_delta_T(T,i)
			T = T + T_delta
			(self.T_all).append(T.array)

		return

	def generate_animation(self,image_path):

		def animate(k):
		# k from 0 to Nt
			plt.clf()

			plt.title("Temperature at t = %d unit time"%(k))
			plt.xlabel("x")
			plt.ylabel("y")

			C = self.T_all[k].reshape(self.N,self.N)

			# This is to plot u_k (u at time-step k)
			plt.pcolormesh(C, cmap=plt.cm.jet, vmin=0,vmax=4)
			plt.colorbar()
			return plt

		def pics():
		# k from 0 to Nt
			for k in range(self.Nt+1):
				if k%5 == 0:
					plt.clf()

					plt.title("Temperature at t = %d unit time"%(k))
					plt.xlabel("x")
					plt.ylabel("y")

					C = self.T_all[k].reshape(self.N,self.N)

					# This is to plot u_k (u at time-step k)
					plt.pcolormesh(C, cmap=plt.cm.jet, vmin=0,vmax=4)
					plt.colorbar()
					plt.savefig(image_path+'/images/T=%d'%(k))

			return

		anim = FuncAnimation(plt.figure(), animate, interval=500, frames=self.Nt+1, repeat=False)
		anim.save(image_path+'/heat_diffusion_simulation.gif', writer='imagemagick', fps=60)
		pics()

		return

	def add_noise(self,noiselevel,image_path):
		self.noiselevel = noiselevel
		gamma = noiselevel*torch.eye(self.N**2)
		self.gamma = gamma

		observations = np.concatenate(self.T_all).reshape(self.Nt+1,self.N**2)
		observations = torch.from_numpy(observations)
		noise = torch.distributions.MultivariateNormal(torch.zeros(self.N**2),gamma).sample(sample_shape=[self.Nt+1])
		observations = observations + noise
		observations = observations.t()
		self.observations = observations.type(torch.FloatTensor)

		def animate(k):
		# k from 0 to Nt
			plt.clf()

			plt.title("Temperature at t = %d unit time"%(k))
			plt.xlabel("x")
			plt.ylabel("y")

			C = self.observations[:,k].reshape(self.N,self.N)

			# This is to plot u_k (u at time-step k)
			plt.pcolormesh(C, cmap=plt.cm.jet, vmin=0,vmax=4)
			plt.colorbar()
			return plt

		anim = FuncAnimation(plt.figure(), animate, interval=500, frames=self.Nt+1, repeat=False)
		anim.save(image_path+'/observations.gif', writer='imagemagick', fps=60)
		
		def pics():
		# k from 0 to Nt
			for k in range(self.Nt+1):
				if k%5 == 0:
					plt.clf()

					plt.title("Temperature at t = %d unit time"%(k))
					plt.xlabel("x")
					plt.ylabel("y")

					C = self.observations[:,k].reshape(self.N,self.N)

					# This is to plot u_k (u at time-step k)
					plt.pcolormesh(C, cmap=plt.cm.jet, vmin=0,vmax=4)
					plt.colorbar()
					plt.savefig(image_path+'/observations/T=%d'%(k))

			return
		pics()
		return
