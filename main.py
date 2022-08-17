from src.simulation.set_up_model import HeatModel2D
from src.model.EnKFmodel import EnKFmodel
from src.model.InverseEnKF import InverseEnKFmodel
import numpy as np
import torch

if __name__ == "__main__":
	
	image_path = 'src/results'# path to save the images

	N = 50
	Nt = 100
	mathcal_K = 0.2
	a,b,c = 25,20,-0.015
	# a,b,c = 25,20,0
	Q_in = 0

	h = 1/(N-1)
	ht = 1/Nt
	r = (mathcal_K*ht)/(a * (h**2))
	r2 = (c*ht)/(a*(h**2))

	noiselevel = 0.1

	print('1. Setup the model with (x:%d,t:%d) data and level of noise %e\n'%(N,Nt,noiselevel))

	# create application context
	# and PETSc nonlinear solver

	# mode: 0 for constant r/r2, 1 for changing r/r2(input is numpy array)
	mode = 1 
	if mode == 1:
		r = np.linspace(1,1.25,Nt+1)*r
		r2 = np.linspace(1,1.5,Nt+1)*r2

	heatmod = HeatModel2D(N, Nt, mode,r,r2)
	heatmod.check_numerical()
	heatmod.compute_all_T(customize=False,T0=None)
	heatmod.generate_animation(image_path+'/simulation')
	if mode == 0:
		print("simulation done! with r : %.3f, r2 : %.3f"%(heatmod.r,heatmod.r2))
	else:
 		print("simulation done! changing r : %.3f, r2 : %.3f"%(np.mean(heatmod.r),np.mean(heatmod.r2)))
	heatmod.add_noise(noiselevel,image_path+'/simulation')
	print('noise added, with noiselevel= %f'%(noiselevel))

	# set up ensemble
	# stopping = 'discrepancy'
	stopping = 'none'
	if mode==0:
		model = EnKFmodel(heatmod,stopping,image_path)
		ensembleSize = 50
		initEnsemble = 'Random'
		print('Ensemble size = %d. Setup the initial ensembles using the %s initialization...\n'%(ensembleSize,initEnsemble))
		model.set_up_ensemble(ensembleSize)
		# with timer.Timer('EnKF timer'):
		model.update_ensemble()
		model.temperature_predict()
	elif mode==1:
		model = InverseEnKFmodel(heatmod,stopping,image_path)
		ensembleSize = 50
		initEnsemble = 'Random'
		print('Ensemble size = %d. Setup the initial ensembles using the %s initialization...\n'%(ensembleSize,initEnsemble))
		model.set_up_ensemble(ensembleSize)
		# with timer.Timer('EnKF timer'):
		model.update_ensemble()
		model.temperature_predict()

	
