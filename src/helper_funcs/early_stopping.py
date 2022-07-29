
def early_stopping(stopping,i,Ri,Ri1,Ei):

	#Discrepancy principle
	if stopping == 'discrepancy':
		if Ri <= 1e-4:
			print('stopping early by '+stopping+'at %d-th iteration'%(i+1))
			return True
		elif Ei <= 1e-5:
			print('stopping early by '+stopping+'at %d-th iteration'%(i+1))
			return True
		else:
			return False
	#Relative error
	if stopping == 'relative':
		tol = 1e-3
		if i > 1:
			if abs(Ri-Ri1) < tol:
				print('stopping early by '+stopping+'at %d-th iteration'%(i+1))
				return True
			else:
				return False
	#None
	if stopping == 'none':
		return False