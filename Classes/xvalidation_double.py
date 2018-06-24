import pandas as pd;
import numpy as np;
from copy import deepcopy;
from sklearn.metrics import accuracy_score,roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve;
from sklearn.metrics import auc as auuc;
import pickle;
from sklearn.linear_model import SGDClassifier;
from sklearn.neighbors import NearestNeighbors;
from sklearn.svm import SVC;from copy import deepcopy;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.tree import DecisionTreeClassifier;
import datetime;

class xvalidation:
	def __init__(self, dataset, probs):
		self.nfolds=7;
		self.curr_classifier=[];
		self.classifiers=[[],[],[],[],[],[]];

		#Indexes: 0-original, 1-oversample, 2-oversample_bubble, 3-oversample_bubble_original, 4-smote
		self.original_training_set=dataset.original_training_set; #Normal split training set 
		self.original_test_set=dataset.original_test_set;#Normal split test set
		self.oversample_tt_set=dataset.oversample_tt_set; #Random oversampling of the minority class 
		self.oversample_bubble_tt_set=dataset.oversample_bubble_tt_set; #Oversampling is only from the bubble set, take all normal instance
		self.oversample_bubble_original_tt_set=dataset.oversample_bubble_original_tt_set #Oversample overall with bubble methods
		self.smote_bubble_tt_set=dataset.smote_bubble_tt_set; #Oversample overall dataset with smote (scikit method)
		self.all_bubbles_tt_set=dataset.all_bubbles_tt_set;

		self.accuracies=[[],[],[],[],[],[]];
		self.precisions0=[[],[],[],[],[],[]];
		self.precisions1=[[],[],[],[],[],[]];
		self.recalls0=[[],[],[],[],[],[]];
		self.recalls1=[[],[],[],[],[],[]];
		self.prs=[[],[],[],[],[],[]];
		self.prAucs=[[],[],[],[],[],[]];
		self.f1scores0=[[],[],[],[],[],[]];
		self.f1scores1=[[],[],[],[],[],[]];
		self.aucs=[[],[],[],[],[],[]];
		self.aucsProbs0=[[],[],[],[],[],[]];
		self.aucsProbs1=[[],[],[],[],[],[]];

		#Indexes: 0-Accuracy, 1-Precision0, 2-Precision1, 3-Recall0, 4-Recall1, 5-PR Auc, 6-f1score0, 7-f1score1, 8-AUC, 9-AUCProbs
		self.scores={'original':[], 'oversample':[], 'oversample_bubble':[], 'oversample_bubble_original':[], 'smote':[], 'allbubbles':[]};
		self.stds={'original':[], 'oversample':[], 'oversample_bubble':[], 'oversample_bubble_original':[], 'smote':[], 'allbubbles':[]};
		self.name_datasets=['original', 'oversample','oversample_bubble', 'oversample_bubble_original', 'smote','allbubbles']
		self.av_accuracy=[0, 0, 0, 0, 0, 0];
		self.av_precision0=[0, 0, 0, 0, 0, 0];
		self.av_precision1=[0, 0, 0, 0, 0, 0];
		self.av_recall0=[0, 0, 0, 0, 0, 0];
		self.av_recall1=[0, 0, 0, 0, 0, 0];
		self.av_prAucs=[0, 0, 0, 0, 0, 0];
		self.av_f1score0=[0, 0, 0, 0, 0, 0];
		self.av_f1score1=[0, 0, 0, 0, 0, 0];
		self.av_auc=[0, 0, 0, 0, 0, 0];
		self.av_aucProbs0=[0, 0, 0, 0, 0, 0];
		self.av_aucProbs1=[0, 0, 0, 0, 0, 0];
		self.std_accuracy=[0, 0, 0, 0, 0, 0];
		self.std_precision0=[0, 0, 0, 0, 0, 0];
		self.std_precision1=[0, 0, 0, 0, 0, 0];
		self.std_recall0=[0, 0, 0, 0, 0, 0];
		self.std_recall1=[0, 0, 0, 0, 0, 0];
		self.std_prAucs=[0, 0, 0, 0, 0, 0];
		self.std_f1score0=[0, 0, 0, 0, 0, 0];
		self.std_f1score1=[0, 0, 0, 0, 0, 0];
		self.std_auc=[0, 0, 0, 0, 0, 0];
		self.std_aucProbs0=[0, 0, 0, 0, 0, 0];
		self.std_aucProbs1=[0, 0, 0, 0, 0, 0];

		self.best_accuracy=[0, 0, 0, 0, 0, 0];
		self.best_precision0=[0, 0, 0, 0, 0, 0];
		self.best_precision1=[0, 0, 0, 0, 0, 0];
		self.best_recall0=[0, 0, 0, 0, 0, 0];
		self.best_recall1=[0, 0, 0, 0, 0, 0];
		self.best_prAucs=[0, 0, 0, 0, 0, 0];
		self.best_f1score0=[0, 0, 0, 0, 0, 0];
		self.best_f1score1=[0, 0, 0, 0, 0, 0];
		self.best_auc=[9999999999999999,9999999999999999,9999999999999999,9999999999999999,9999999999999999,9999999999999999];
		self.best_aucProbs0=[9999999999999999,9999999999999999,9999999999999999,9999999999999999,9999999999999999,9999999999999999];
		self.best_aucProbs1=[9999999999999999,9999999999999999,9999999999999999,9999999999999999,9999999999999999,9999999999999999];
		self.best_classifiers=[[],[],[],[],[],[]];
		self.y_predicts=[[],[],[],[],[],[]]

		self.probs=probs;

	def set_classifier(self, classifier):
		self.curr_classifier=classifier;

	def get_average_scores(self):
		for i in range(0, 6):
			##Compute Accuracies
			self.av_accuracy[i]=np.mean(self.accuracies[i]);
			self.av_precision0[i]=np.mean(self.precisions0[i]);
			self.av_precision1[i]=np.mean(self.precisions1[i]);
			self.av_recall0[i]=np.mean(self.recalls0[i]);
			self.av_recall1[i]=np.mean(self.recalls1[i]);
			self.av_prAucs[i]=np.mean(self.prAucs[i]);
			self.av_f1score0[i]=np.mean(self.f1scores0[i]);
			self.av_f1score0[i]=np.mean(self.f1scores1[i]);
			self.av_auc[i]=np.mean(self.aucs[i]);
			self.av_aucProbs0[i]=np.mean(self.aucsProbs0[i]);
			self.av_aucProbs1[i]=np.mean(self.aucsProbs1[i]);

			##Compute Stadard Deviations
			self.std_accuracy[i]=np.std(self.accuracies[i]);
			self.std_precision0[i]=np.std(self.precisions0[i]);
			self.std_precision1[i]=np.std(self.precisions1[i]);
			self.std_recall0[i]=np.std(self.recalls0[i]);
			self.std_recall1[i]=np.std(self.recalls1[i]);
			self.std_prAucs[i]=np.std(self.prAucs[i]);
			self.std_f1score0[i]=np.std(self.f1scores0[i]);
			self.std_f1score0[i]=np.std(self.f1scores1[i]);
			self.std_auc[i]=np.std(self.aucs[i]);
			self.std_aucProbs0[i]=np.std(self.aucsProbs0[i]);
			self.std_aucProbs1[i]=np.std(self.aucsProbs1[i]);

			print(self.name_datasets[i])
			print('          ',i,'_ : Accuracy-', self.av_accuracy[i], ', Precicion 0-', self.av_precision0[i], ', Precision 1-', self.av_precision1[i], ', Recall 1-', self.av_recall1[i], ', Recall 0-', self.av_recall0[i], ', PR Aucs-', self.prAucs[i], ', F1 Score 0-', self.av_f1score0[i], ', F1 Score1-', self.av_f1score1, ', AUC-', self.av_auc[i], ', AUC Probs 0-', self.av_aucProbs0[i], ', AUC Probs 1-', self.av_aucProbs1);

			self.scores[self.name_datasets[i]] = [self.av_accuracy[i], self.av_precision0[i], self.av_precision1[i], self.av_recall0[i], self.av_recall1[i], self.av_prAucs[i],self.av_f1score0[i], self.av_f1score1[i], self.av_auc[i], self.av_aucProbs0[i], self.av_aucProbs1[i]]
			self.stds[self.name_datasets[i]] = [self.std_accuracy[i], self.std_precision0[i], self.std_precision1[i], self.std_recall0[i], self.std_recall1[i], self.std_prAucs[i],self.std_f1score0[i], self.std_f1score1[i], self.std_auc[i], self.std_aucProbs0[i], self.std_aucProbs1[i]]

	def cross_validate_all(self):
		#0General
		t0=datetime.datetime.now().replace(microsecond=0);
		print("   ORIGINAL TRAINING SET");
		self.cross_validate_one(self.original_training_set, 0);
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('     Time taken TO TRAIN THE ORIGINAL TRAINING SET: ', tnow-t0)
		t0=tnow;
		#1Random Oversample
		print("   RANDOM OVERSAMPLE TRAINING SET");
		self.cross_validate_one(self.smotebubble_tt_set, 1);
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('     Time taken TO TRAIN THE RANDOM OVERSAMPLING TRAINING SET: ', tnow-t0)
		t0=tnow;
		#2Bubble Oversample
		print("   RANDOM OVERSAMPLE ON THE BUBBLE TRAINING SET");
		self.cross_validate_one(self.oversample_bubble_tt_set, 2);
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('     Time taken TO TRAIN THE RANDOM OVERSAMPLING ON THE BUBBLE TRAINING SET: ', tnow-t0)
		t0=tnow;
		#3Bubble Original Oversample
		print("   RANDOM OVERSAMPLE ON THE BUBBLE AND ORIGINAL TRAINING SET");
		self.cross_validate_one(self.oversample_bubble_original_tt_set, 3);
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('     Time taken TO TRAIN THE RANDOM OVERSAMPLING ON THE BUBBLE AND ORIGINAL TRAINING SET: ', tnow-t0)
		t0=tnow;
		#4Smote Oversample
		print("   RANDOM OVERSAMPLE ON THE SMOTE TRAINING SET");
		self.cross_validate_one(self.smote_bubble_tt_set, 4);
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('     Time taken TO TRAIN THE RANDOM OVERSAMPLING ON THE SMOTE TRAINING SET: ', tnow-t0)
		t0=tnow;
		print("   ENTIRE BUBBLE SET");
		self.cross_validate_one(self.all_bubbles_tt_set, 5);
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('     Time taken TO TRAIN THE RANDOM OVERSAMPLING ON THE ALL BUBBLE TRAINING SET: ', tnow-t0)
		t0=tnow;


	def cross_validate_one(self, dataset, index):
		clf = deepcopy(self.curr_classifier);
		for fold in range(0, self.nfolds):
			print('        ',fold)
			clf.fit(dataset['xs'][fold], dataset['ys'][fold]);
			ypredict=clf.predict(self.original_test_set['xs'][fold]);
			ypredictProbs = 0;
			if self.probs:
				ypredictProbs=clf.predict_proba(self.original_test_set['xs'][fold])
			best = self.compute_scores(ypredict, ypredictProbs, self.original_test_set['ys'][fold], index);
			if best:
				self.best_classifiers[index]=clf;
			self.classifiers[index].append(clf);


	def compute_scores(self, ypredict, ypredProbs, yTrue, index):
		best = False;
		accuracy = accuracy_score(yTrue, ypredict)
		self.accuracies[index].append(accuracy);

		precision0 = precision_score(yTrue, ypredict, 0);
		precision1 = precision_score(yTrue, ypredict, 1);
		self.precisions0[index].append(precision0);
		self.precisions1[index].append(precision1);

		recall0 = recall_score(yTrue, ypredict, 0);
		recall1 = recall_score(yTrue, ypredict, 1);
		self.recalls0[index].append(recall0);
		self.recalls1[index].append(recall1);

		precision, recall, thresholds = precision_recall_curve(yTrue, ypredProbs[:,1], pos_label=1)
		prAuc = auuc(recall, precision);
		self.prAucs[index].append(prAuc);

		f1score0 = f1_score(yTrue, ypredict, 0);
		f1score1 = f1_score(yTrue, ypredict, 1);
		
		self.f1scores0[index].append(f1score0);
		self.f1scores1[index].append(f1score1);

		auc = roc_auc_score(yTrue, ypredict);
		self.aucs[index].append(auc);

		aucProbs0=0;
		aucProbs1=0;
		if self.probs:
			aucProbs0 = roc_auc_score(yTrue, ypredProbs[:,0]);
			self.aucsProbs0[index].append(aucProbs0);
			aucProbs1 = roc_auc_score(yTrue, ypredProbs[:,1]);
			self.aucsProbs1[index].append(aucProbs1);

		if f1score0 > self.best_f1score0[index]:
			best=True
			self.best_accuracy[index] = accuracy;
			self.best_precision0[index] = precision0;
			self.best_precision1[index] = precision1;
			self.best_recall0[index] = recall0;
			self.best_recall1[index] = recall1;
			self.best_prAucs[index] = prAuc;
			self.best_f1score0[index] = f1score0;
			self.best_f1score1[index] = f1score1;
			self.best_auc[index] = auc;

			if self.probs:
				self.best_aucProbs0[index] = aucProbs0;
				self.best_aucProbs1[index] = aucProbs1;

		print('          ','Accuracy: ', accuracy, ', Precicion 0: ', precision0, ', Precision 1', precision1, ', Recall 1: ', recall1, ', Recall 0: ', recall0, ', PR AUC: ', prAuc, ', F1 Score 0: ', f1score0, ', F1 Score1: ', f1score1, ', AUC: ', auc, ', AUC Probs 0: ', aucProbs0, ', AUC Probs 1: ', aucProbs1);

		return best;


	def run(self, classifier):
		self.set_classifier(classifier);

		t0=datetime.datetime.now().replace(microsecond=0);
		print("CROSS VALIDATION");
		self.cross_validate_all();
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('  Time taken to cross validate everything: ', tnow-t0)
		t0=tnow;

		print("GETTING AVERAGE SCORES");
		self.get_average_scores();
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('  Time taken to get all the average scores: ', tnow-t0)
		t0=tnow;

		print('Summary!!')
		(pd.DataFrame(self.scores))
		
		








