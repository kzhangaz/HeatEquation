from src.model import EnKF

if __name__ == "__main__":
	
	# choose method: 1 for EnKF, 5 for meanfield
	method = 1
	image_path = 'src/results/images'# path to save the images

	# set up model
	N = 50 # number of steps
	Nt = 50 # number of steps in time
	noiselevel = 0.01**2
	# noiselevel = 0

	print('1. Setup the model with (x:%d,t:%d) data and level of noise %.3f\n'%(N,Nt,noiselevel))

	model = EnKF.EnKFmodel(N,Nt,noiselevel)
	model.set_up_model(image_path)

	# set up ensemble
	ensembleSize = 200

	#initEnsemble = 'KL'; % Karhunen-Loeve expansion
	#initEnsemble = 'random'; % Normally distributed around the mean of uexact
	initEnsemble = 'random'

	print('3. Ensemble size = %d. Setup the initial ensembles using the %s initialization...\n'%(ensembleSize,initEnsemble))

	model.set_up_ensemble(ensembleSize,initEnsemble)

	stopping = 'relative' # 'relative' or 'discrepancy'
	method = 1
	model.update_model(method,image_path,stopping)
	
	
