from pickle import TRUE
from torch import mm,matmul
from torch import linalg
from torch import mean,FloatTensor
from torch import isreal,real,zeros,multinomial,linspace,t,flatten
from src import covmat
from src import moments
from src.update import update_EnKF
from src.vecmul import vecmul

def update_mean_field(self,maxit,stopping,Minteracting,tfin,image_path):
	print("running update_mean_field...")

	time = []
	time.append(0)

	radius = []

	for i in range(maxit):

		eig1 = mm((self.m2-vecmul(self.m1,self.m1)),t(self.G))
		eig2 = mm(linalg.pinv(self.gamma),self.G)
		eig,_ = linalg.eig(-mm(eig1,eig2))

		if mean(isreal(eig).type(FloatTensor)) != 1:
			eig = real(eig)

		if self.ensembleSize==Minteracting and i==0:
			spectrum = abs(eig)

		radius.append(max(abs(eig)))
		dt = 1/radius[i]

		if time[i]+dt > tfin:
			dt = tfin-time[i]
		if time[i] >= tfin:
			break

		self.convergence()

		if i > 0:
			if update_EnKF.early_stopping(stopping,i,self.M[i],self.M[i-1],self.noise):
				break
		
		if Minteracting != self.ensembleSize:
			TempEn = zeros(self.En.size())

		Gu = flatten(self.predict_by_En(self.En),start_dim=0,end_dim=1)

		for j in range(self.ensembleSize):
			# xi = noiseLevel.*randn(1,N) # mvnrnd(zeros(1,N),gamma)
			part2 = matmul(linalg.inv(self.gamma),\
				flatten(self.observations,start_dim=0,end_dim=1) - Gu[:,j])

			if Minteracting == self.ensembleSize:
				grad = dt* matmul(eig1,part2)
				self.En[:,j] = self.En[:,j] + grad # + sqrt(dt)*t(xi);
			else:
				Idx = multinomial(linspace(1,self.ensembleSize,self.ensembleSize),Minteracting)
				m1Idx,m2Idx = moments.moments(self.En[:,Idx])
				part1 = dt* matmul( (m2Idx- vecmul(m1Idx,m1Idx) ) , t(self.G) )
				grad = matmul(part1,part2)
				TempEn[:,j] = self.En[:,j] + grad # + sqrt(dt)*t(xi);
		
		if Minteracting == self.ensembleSize:
			self.m1,self.m2 = moments.moments(self.En)
		else:
			self.En = TempEn
			self.m1,self.m2 = moments.moments(self.En)

		if ((time[i]/tfin) * 100) % 10 == 0:
			print(' %d percent of tfin finished running'%((time[i]/tfin) * 100))
		
		time.append(time[i] + dt)
	
	self.convergence()
	
	self.final_plot(i,image_path,method=5)
	
	return