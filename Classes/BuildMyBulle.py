import pandas as pd;
import numpy as np;
import datetime;
import pickle;
import Dataset_Double;
import xvalidation;
from IPython.display import display;
from sklearn.preprocessing import StandardScaler
from types import MethodType
from sklearn.linear_model import SGDClassifier;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.svm import SVC;from copy import deepcopy;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.tree import DecisionTreeClassifier;
import Generate_Data;
import Run_imbalances;
import Imbalance_Plotter;
class BuildMyBulle:
	def __init__(self, categories, datasetLabel, name_experiment, bulle,bubble_oversample):
		self.DummyDataset=[];
		self.name_experiment=name_experiment;
		self.datasetLabel=datasetLabel;
		self.datasetNames=['haberman', 'pulsar', 'magic', 'tictactoe', 'Maxi', 'Midi', 'Mini', 'Tiny', 'Micro', 'Nano'];
		self.categories=categories;
		self.bulle=bulle;
		self.bubble_oversample=bubble_oversample;
		self.xval_dt=[];
		self.xval_rf=[];
		self.xval_knn=[];
		self.xval_sgd=[];
		self.xval_svc=[];

	def scale(self,attributes):
		scaler = StandardScaler();
		return pd.DataFrame(scaler.fit_transform(attributes)), scaler;

	def save_xvalidation(self, obj, path):
		with open(path, 'wb') as fp:
			pickle.dump(obj, fp)

	def set_instantiate_data(self, instantiate_data_fct):
		self.instantiate_data= MethodType(instantiate_data_fct, self)

	def instantiate_data(self):
		data_generator = Generate_Data.Generate_Data(self.datasetLabel, self.bulle, self.bubble_oversample, self.categories, self.name_experiment);


		if self.datasetLabel==0:
			data_generator.instantiate_haberman();
		elif self.datasetLabel==1:
			data_generator.instantiate_pulsar();
		elif self.datasetLabel==2:
			data_generator.instantiate_magic();
		elif self.datasetLabel==3:
			data_generator.instantiate_tictactoe();

		if self.datasetLabel <= 3:
			self.DummyDataset=data_generator.DummyDataset;

		if self.datasetLabel>3:
			data_generator.instantiate_imbalances();



	def cross_validate(self):
		if self.datasetLabel<=3:

			path = '../Algorithms_Scores/experimentdouble_'+self.datasetNames[self.datasetLabel]+'_'+self.name_experiment;
			#Stochastic Gradient Descent
			clf = SGDClassifier(loss='log', tol=0.00001);
			xvalidation_sgd = xvalidation.xvalidation(self.DummyDataset, True);
			xvalidation_sgd.run(clf)
			temppath = path+'_sgd.txt'
			self.save_xvalidation(xvalidation_sgd, temppath);
			self.xval_sgd=xvalidation_sgd;


			#Nearest Neighbours
			clf = KNeighborsClassifier();
			xvalidation_knn = xvalidation.xvalidation(self.DummyDataset, True);
			xvalidation_knn.run(clf)
			temppath = path+'_knn.txt'
			self.save_xvalidation(xvalidation_knn, temppath);
			self.xval_knn=xvalidation_knn;

			#Support Vector Machine Classifier
			clf = SVC(probability=True);
			xvalidation_svc = xvalidation.xvalidation(self.DummyDataset, True);
			xvalidation_svc.run(clf)
			temppath = path+'_svc.txt';
			self.save_xvalidation(xvalidation_svc, temppath);
			self.xval_svc=xvalidation_svc;

			#Random Forest
			clf = RandomForestClassifier();
			xvalidation_rf = xvalidation.xvalidation(self.DummyDataset, True);
			xvalidation_rf.run(clf)
			temppath = path+'rf.txt';
			self.save_xvalidation(xvalidation_rf, temppath);
			self.xval_rf=xvalidation_rf;

			#Decision Tree Classifier
			clf = DecisionTreeClassifier();
			xvalidation_dt = xvalidation.xvalidation(self.DummyDataset, True);
			xvalidation_dt.run(clf)
			temppath = path+'_dt.txt';
			self.save_xvalidation(xvalidation_dt, temppath);
			self.xval_dt=xvalidation_dt;

		else:
			run_imb = Run_imbalances.Run_imbalances(self.datasetNames[self.datasetLabel], self.name_experiment);
			run_imb.run();
			self.DummyDataset=run_imb;


	def visualize_scores(self):
		if self.datasetLabel <=3:
			names_scores=['Average Precision', 'Precision 0 ', 'Precision 1', 'Recall 0', 'Recall 1', 'PRAUC', 'F1 Score 0', 'F1 Score 1', 'AUROC', 'AUC Probs 0', 'AUC Probs 1'];
			print('Decision Tree')
			display(pd.DataFrame(self.xval_dt.scores, index=names_scores));
			print('K Nearest Neighbour')
			display(pd.DataFrame(self.xval_knn.scores, index=names_scores));
			print('Random Forest')
			display(pd.DataFrame(self.xval_rf.scores, index=names_scores));
			print('Stochastic Gradient Descent')
			display(pd.DataFrame(self.xval_sgd.scores, index=names_scores));
			print('Support Vector Machine')
			display(pd.DataFrame(self.xval_svc.scores, index=names_scores));
		else:
			print(self.datasetNames[self.datasetLabel]);
			for i in range(0, len(self.DummyDataset.xval_sgd)):
				print('SIZE', self.DummyDataset.files[i]);
				names_scores=['Average Precision', 'Precision 0 ', 'Precision 1', 'Recall 0', 'Recall 1', 'PRAUC', 'F1 Score 0', 'F1 Score 1', 'AUROC', 'AUC Probs 0', 'AUC Probs 1'];
				print('Decision Tree')
				display(pd.DataFrame(self.DummyDataset.xval_dt[i].scores, index=names_scores));
				print('K Nearest Neighbour')
				display(pd.DataFrame(self.DummyDataset.xval_knn[i].scores, index=names_scores));
				print('Random Forest')
				display(pd.DataFrame(self.DummyDataset.xval_rf[i].scores, index=names_scores));
				print('Stochastic Gradient Descent')
				display(pd.DataFrame(self.DummyDataset.xval_sgd[i].scores, index=names_scores));
				print('Support Vector Machine')
				display(pd.DataFrame(self.DummyDataset.xval_svc[i].scores, index=names_scores));
			imb_plotter_tiny = Imbalance_Plotter.Imbalance_Plotter(self.datasetNames[self.datasetLabel]);
			imb_plotter_tiny.initialise_score_dics();
			imb_plotter_tiny.process_all_scores(self.DummyDataset);
			imb_plotter_tiny.run_all_performances(self.datasetNames[self.datasetLabel])

		
    

