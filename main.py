from src.simulation.set_up_model import HeatModel2D

if __name__ == "__main__":
	
	# choose method: 1 for EnKF, 5 for meanfield
	method = 1
	image_path = 'src/results/images'# path to save the images

	# set up model
	N = 10 # number of steps
	Nt = 50 # number of steps in time
	# Nt = 30
	noiselevel = 0.01**2
	# noiselevel = 0

	print('1. Setup the model with (x:%d,t:%d) data and level of noise %e\n'%(N,Nt,noiselevel))

	# model = EnKF.EnKFmodel(N,Nt,noiselevel)
	# OptDB = PETSc.Options()

	# # initialize the parameters
	# N = OptDB.getInt('N', 50)
	# Nt = OptDB.getInt('Nt', 100)
	# mathcal_K = OptDB.getReal('mathcal_K', 0.2)
	# a = OptDB.getReal('a',25)
	# b = OptDB.getReal('b',20)
	# c = 0
	# Q_in = 0

	N = 50
	Nt = 100
	mathcal_K = 0.2
	a,b,c = 25,20,0
	Q_in = 0
	# create application context
	# and PETSc nonlinear solver
	heatmod = HeatModel2D(N, Nt, mathcal_K, a, b, c, Q_in)
	heatmod.check_numerical()
	heatmod.compute_all_T()
	heatmod.generate_animation(image_path+'/simulation')


	# set up ensemble
	ensembleSize = 200

	#initEnsemble = 'KL'; % Karhunen-Loeve expansion
	#initEnsemble = 'random'; % Normally distributed around the mean of uexact
	initEnsemble = 'random'

	print('3. Ensemble size = %d. Setup the initial ensembles using the %s initialization...\n'%(ensembleSize,initEnsemble))

	model.set_up_ensemble(ensembleSize,initEnsemble)

	stopping = 'discrepancy' # 'relative' or 'discrepancy'
	method = 1
	model.update_model(method,image_path,stopping)
	
	
