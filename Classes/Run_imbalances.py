import pandas as pd;
import numpy as np;
from matplotlib import pyplot as plt; 
import pickle;
from sklearn.linear_model import SGDClassifier;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.svm import SVC;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.tree import DecisionTreeClassifier;
import xvalidation;
import Dataset_Double

class Run_imbalances:
	def __init__(self, name_dataset, name_experiment):
		self.name_dataset=name_dataset;
		self.datasets=[];		
		self.xval_sgd = [];
		self.xval_knn=[];
		self.xval_svc=[];
		self.xval_rf=[];
		self.xval_dt=[];
		self.files=['Fifty', 'Fifty5', 'Sixty', 'Sixty5', 'Seventy', 'Seventy5', 'Eighty', 'Eighty5', 'Ninety', 'Ninety5'];
		self.name_experiment=name_experiment;

	def read_datasets(self):
		for file in self.files:
			path='../DataSets/Unbalanced/'+self.name_experiment+self.name_dataset+'_'+file+'.txt';
			with open(path, 'rb') as fp:
				self.datasets.append(pickle.load(fp));


	def save_xvalidation(self, xval, path):
		with open(path, 'wb') as fp:
			pickle.dump(xval, fp);


	def get_all_scores(self, dataset, file):
		#Stochastic Gradient Descent
		clf = SGDClassifier(loss='log', tol=0.00001);
		xvalidation_sgd = xvalidation.xvalidation(dataset, True);
		xvalidation_sgd.run(clf)
		path='../Algorithms_Scores/'+self.name_experiment+self.name_dataset+'_'+file+'_sgd.py'
		self.save_xvalidation(xvalidation_sgd, path)

		#K Nearest Neighbours
		clf = KNeighborsClassifier();
		xvalidation_knn = xvalidation.xvalidation(dataset, True);
		xvalidation_knn.run(clf)
		path='../Algorithms_Scores/'+self.name_experiment+self.name_dataset+'_'+file+'_knn.py'
		self.save_xvalidation(xvalidation_knn, path)

		#Support Vector Machine
		clf = SVC(probability=True);
		xvalidation_svc = xvalidation.xvalidation(dataset, True);
		xvalidation_svc.run(clf)
		path='../Algorithms_Scores/'+self.name_experiment+self.name_dataset+'_'+file+'_svc.py'
		self.save_xvalidation(xvalidation_svc, path)

		#Random Forest
		clf = RandomForestClassifier();
		xvalidation_rf = xvalidation.xvalidation(dataset, True);
		xvalidation_rf.run(clf)
		path='../Algorithms_Scores/'+self.name_experiment+self.name_dataset+'_'+file+'_rf.py'
		self.save_xvalidation(xvalidation_rf, path)

		#Decision Tree Classifier 
		clf = DecisionTreeClassifier();
		xvalidation_dt = xvalidation.xvalidation(dataset, True);
		xvalidation_dt.run(clf);
		path='../Algorithms_Scores/'+self.name_experiment+self.name_dataset+'_'+file+'_dt.py'
		self.save_xvalidation(xvalidation_dt, path)

		self.xval_sgd.append(xvalidation_sgd);
		self.xval_knn.append(xvalidation_knn);
		self.xval_svc.append(xvalidation_svc);
		self.xval_rf.append(xvalidation_rf);
		self.xval_dt.append(xvalidation_dt);

	def get_all_algorithms_scores(self):
		for dataset in range(0, len(self.datasets)):
			self.get_all_scores(self.datasets[dataset], self.files[dataset]);

	def run(self):
		self.read_datasets();
		self.get_all_algorithms_scores();





