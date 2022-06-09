from unicodedata import name
from src.HeatEquation1D import control_func1 as cf
from src.model import EnKF

if __name__ == "__main__":
	
	# choose method: 1 for EnKF, 5 for meanfield
	method = 1
	# a = 0; b = math.pi

	test = 0 # choose the continuous control function
	image_path = 'src/HeatEquation1D/images'# path to save the images

	control_func = cf.control_func(test)
	sol_func = None

	# set up model
	N = 2**8 # number of steps
	Nt = 100 # number of steps in time
	noiselevel = 0.01**2

	print('1. Setup the model with (x:%d,t:%d) data and level of noise %f\n'%(N,Nt,noiselevel))

	thename = 'HeatEquation1D'
	model = EnKF.EnKFmodel(thename,N*Nt,N,control_func,noiselevel,sol_func)
	model.set_up_model(image_path)

	# set up ensemble
	ensembleSize = 200

	#initEnsemble = 'KL'; % Karhunen-Loeve expansion
	#initEnsemble = 'random'; % Normally distributed around the mean of uexact
	initEnsemble = 'brownian'

	print('3. Ensemble size = %d. Setup the initial ensembles using the %s initialization...\n'%(ensembleSize,initEnsemble))

	model.set_up_ensemble(ensembleSize,initEnsemble)

	model.update_model(method,image_path)
	
	
