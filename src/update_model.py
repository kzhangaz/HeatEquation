from src import timer
from src.update import update_EnKF
from src.update import update_mean_field

def update_model(self,method,image_path,stopping):

	if method == 1:
		print('4. EnMKF solver')
		with timer.Timer('EnKF timer'):
			update_EnKF.update_EnKF(self,stopping,image_path)

	# if method == 5:
	# 	Minteracting = self.ensembleSize
	# 	tfin = 10
	# 	with timer.Timer('mean_field timer'):
	# 		update_mean_field.update_mean_field(self,maxit,stopping,Minteracting,tfin,image_path)
