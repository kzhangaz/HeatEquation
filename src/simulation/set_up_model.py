from __future__ import division
import sys, petsc4py
import numpy as np
import matplotlib.pyplot as plt

petsc4py.init(sys.argv)

from petsc4py import PETSc
from matplotlib.animation import FuncAnimation

class HeatModel2D:

	def __init__(self, N, Nt, mathcal_K, a, b, c, Q_in):
		self.N = N
		self.Nt = Nt
		h = 1/N
		ht = 1/Nt
		self.mathcal_K = mathcal_K
		self.a = a
		self.b = b
		self.c = c
		self.Q_in = Q_in
		self.r = (mathcal_K*ht)/(a * (h**2))
		self.T_all = []

	def check_numerical(self):
		if 1-4*(self.r) <= 0:
			raise ValueError("Numerically Not Stable!")
		else:
			print("Numerically OK! ;)")

	def get_X_all(self, X, Xn, Xs, Xw, Xe):

		X = X.reshape((self.N-1,self.N-1))
		X_all = np.concatenate((Xn[np.newaxis,:], X, Xs[np.newaxis,:]), axis=0)
		X_all = np.concatenate((Xw[:,np.newaxis], X_all, Xe[:,np.newaxis]), axis=1)

		return X_all

	def compute(self, X, Xn, Xs, Xw, Xe):

		# Create a new sparse PETSc matrix, fill it and then assemble it
		l = (self.N-1)**2
		r = self.r
		A = PETSc.Mat().createAIJ([l, l])
		A.setUp()

		diagonal_entry = 1 - 4*r
		off_diagonal_entry = r

		diag = PETSc.Vec().createSeq(l)
		diag.setArray(diagonal_entry*np.ones(l))
		A.setDiagonal(diag)

		for i in range(l):
			if i>0: 
				if i%(self.N-1) != 0:
					A.setValue(i, i-1, off_diagonal_entry)
				if i-(self.N-1) >= 0:
					A.setValue(i, i-(self.N-1), off_diagonal_entry)
			if i<(l-1):
				if i%(self.N-1) != 2:
					A.setValue(i, i+1, off_diagonal_entry)
				if i+self.N-1 <= l-1:
					A.setValue(i, i+(self.N-1), off_diagonal_entry)

		A.assemble()

		# # print to check
		# for i in range(l):
		#     print(A.getValues(i,np.arange(l,dtype=np.int32)))

		# update
		ht = 1/(self.Nt)
		h = 1/(self.N)
		N = self.N
		r2 = ((self.c)*ht)/((self.a)*(h**2))
		bQ = ((self.b)*(self.Q_in))/((self.a)*(self.Nt))
		X_left = X.array.reshape(N-1,N-1).T[0,:]
		# print("X_left:")
		# print(np.mean(X_left))

		Xn_next = (1-3*r)*Xn + r*np.concatenate((0,Xn[1:]),axis=None)\
		+ r*np.concatenate((Xn[:-1],0),axis=None) \
		+ r*(X.array[:N-1]) + bQ
		Xn_next[0] = Xn_next[0] + r*Xw[0]
		Xn_next[-1] = Xn_next[-1] + r*Xe[0]

		Xs_next = (1-3*r)*Xs + r*np.concatenate((0,Xs[1:]),axis=None)\
		+ r*np.concatenate((Xs[:-1],0),axis=None) \
		+ r*(X.array[-(N-1):]) + bQ
		Xs_next[0] = Xs_next[0] + r*Xw[-1]
		Xs_next[-1] = Xs_next[-1] + r*Xe[-1]

		Xw_next = (1-3*r+r2)*Xw + r*np.concatenate((0,Xw[1:]),axis=None)\
		+ r*np.concatenate((Xw[:-1],0),axis=None)\
		+ r*np.concatenate((0,X_left,0),axis=None) + bQ\
		+ r2
		Xw_next[0] = Xw_next[0] + r*Xw[0]
		Xw_next[-1] = Xw_next[-1] + r*Xw[-1]

		Xe_next = Xe

		u_np = np.zeros((N-1,N-1))
		u_np[0,:] = r*Xn
		u_np[-1,:] = r*Xs
		u_np[:,0] = r*Xw[1:-1]
		u_np[:,-1] = r*Xe[1:-1]
		u_np = u_np.ravel()

		X_next = PETSc.Vec().createSeq(l)
		A.mult(X,X_next)
		u = PETSc.Vec().createSeq(l)
		u.setArray(u_np)
		X_next = X_next + u

		return X_next, Xn_next, Xs_next, Xw_next, Xe_next


	def compute_all_T(self):

		#initialize
		N = self.N
		l = (N-1)**2

		X = PETSc.Vec().createSeq(l)
		X.setArray(np.ones(l))
		Xn = np.ones(N-1) # 1 to N-1
		Xs = np.ones(N-1) # 1 to N-1
		Xw = np.ones(N+1) # 0 to N
		Xe = 4*np.ones(N+1) # 0 to N

		X_all = self.get_X_all(X.array, Xn, Xs, Xw, Xe)
		(self.T_all).append(X_all)

		for i in range(self.Nt):
			X, Xn, Xs, Xw, Xe = self.compute(X, Xn, Xs, Xw, Xe)
			X_all = self.get_X_all(X.array, Xn, Xs, Xw, Xe)
			(self.T_all).append(X_all)

		return

	def generate_animation(self,image_path):

		def animate(k):
		# k from 0 to Nt
			plt.clf()

			plt.title("Temperature at t = %d unit time"%(k))
			plt.xlabel("x")
			plt.ylabel("y")

			C = self.T_all[k]

			# This is to plot u_k (u at time-step k)
			plt.pcolormesh(C, cmap=plt.cm.jet, vmin=0,vmax=4)
			plt.colorbar()
			return plt

		def pics(k):
		# k from 0 to Nt
			plt.clf()

			plt.title("Temperature at t = %d unit time"%(k))
			plt.xlabel("x")
			plt.ylabel("y")

			C = self.T_all[k]

			# This is to plot u_k (u at time-step k)
			plt.pcolormesh(C, cmap=plt.cm.jet, vmin=0,vmax=4)
			plt.colorbar()
			if k%5 == 0:
				plt.savefig(image_path+'/pics/T=%d'%(k))
			return plt

		anim1 = FuncAnimation(plt.figure(), pics, interval=500, frames=self.Nt+1, repeat=False)
		anim1.save(image_path+'/hey.gif', writer='imagemagick', fps=60)
		anim = FuncAnimation(plt.figure(), animate, interval=500, frames=self.Nt+1, repeat=False)
		anim.save(image_path+'/heat_equation_simulation.gif', writer='imagemagick', fps=60)

		return


