from src.simulation import set_up_model
from src import set_up_ensemble
from src.update.convergence import convergence
from src import update_model
from src import final_plots

class EnKFmodel(object):
	
	def __init__(self,N,Nt,noiselevel):
		self.Nt = Nt
		self.N = N
		self.noiselevel = noiselevel # scalar

		self.E_theta = []
		self.E_T = []
		self.R_theta = []
		self.R_T = []
		self.M = [] # M[i] size: 1 * 1
		self.theta_hat = []

	set_up_model = set_up_model.set_up_model
	# l,A,T,y,theta,u,noise,gamma are set
	# l = (N-1)^2
	# theta = torch.tensor([mathcal_K,a,b])
	# A: (N-1)^2 * (N-1)^2
	# T: (N-1)^2 * Nt+1
	# y = T + noise
	# u: (N-1)^2 * Nt+1
	# noise size: l * Nt+1
	# gamma (cov of noise distribution):  l * l

	set_up_ensemble = set_up_ensemble.set_up_ensemble
	# En,initEnsemble,ensembleSize,m1,m2 are set
	# ensembleSize: J=200
	# En size: N * J
	# m1: N

	convergence = convergence

	update_model = update_model.update_model

	final_plot = final_plots.final_plot

	
	




	