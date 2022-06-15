from src.simulation import set_up_model
from src import set_up_ensemble
from src.update.convergence import convergence
from src import update_model
from src import final_plots
from src.predict_by_En import predict_by_En

class EnKFmodel(object):
	
	def __init__(self,N,Nt,noiselevel):
		self.Nt = Nt
		self.N = N
		self.noiselevel = noiselevel # scalar

		self.E = []
		self.R = []
		# self.AE = []
		# self.AR = []
		self.M = [] # M[i] size: 1 * 1

	set_up_model = set_up_model.set_up_model
	#A,G,observations,u_exact,p,noise,gamma are set
	# G (y = G*u): K * N
	# p: K
	# observations: K
	# u_all: N * Nt
	# w_exact: N_control * Nt_control
	# noise size: K
	# gamma (cov of noise distribution):  K * K

	set_up_ensemble = set_up_ensemble.set_up_ensemble
	# En,initEnsemble,ensembleSize,m1,m2 are set
	# ensembleSize: J=200
	# En size: N * J
	# m1: N
	# m2: 
	predict_by_En = predict_by_En

	convergence = convergence

	update_model = update_model.update_model

	final_plot = final_plots.final_plot

	
	




	