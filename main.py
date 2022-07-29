from src.simulation.set_up_model import HeatModel2D
from src.model.EnKFmodel import EnKFmodel

if __name__ == "__main__":
	
	image_path = 'src/results'# path to save the images

	N = 50
	Nt = 100
	mathcal_K = 0.2
	a,b,c = 25,20,0
	Q_in = 0

	noiselevel = 0.1

	print('1. Setup the model with (x:%d,t:%d) data and level of noise %e\n'%(N,Nt,noiselevel))

	# create application context
	# and PETSc nonlinear solver
	heatmod = HeatModel2D(N, Nt, mathcal_K, a, b, c, Q_in)
	heatmod.check_numerical()
	heatmod.compute_all_T()
	heatmod.generate_animation(image_path+'/simulation')
	print("simulation done! with r : %.3f, r2 : %.3f"%(heatmod.r,heatmod.r2))
	heatmod.add_noise(noiselevel,image_path+'/simulation')
	print('noise added, with noiselevel= %f'%(noiselevel))

	# set up ensemble
	stopping = 'discrepancy'
	# stopping = 'none'
	model = EnKFmodel(heatmod,stopping,image_path)
	ensembleSize = 50
	initEnsemble = 'Random'
	print('Ensemble size = %d. Setup the initial ensembles using the %s initialization...\n'%(ensembleSize,initEnsemble))
	model.set_up_ensemble(ensembleSize)
	# with timer.Timer('EnKF timer'):
	model.update_ensemble()

	
