import pandas as pd;
import numpy as np;
from matplotlib import pyplot as plt;
from copy import deepcopy
from Run_imbalances import Run_imbalances;
from xvalidation import xvalidation;

class Imbalance_Plotter:
	def __init__(self, imbalance_name):
		self.scores_sgd={};
		self.scores_knn={};
		self.scores_svc={};
		self.scores_rf={};
		self.scores_dt={};
		self.scores=[];
		self.imbalance_name=imbalance_name;
		self.name_datasets=['original', 'oversample', 'oversample_bubble', 'oversample_bubble_original', 'smote'];
		self.name_performances=['Accuracy', 'Precision 0', 'Precision 1', 'Recall 0', 'Recall1', 'Balanced_Accuracy', 'F1 Score 0', 'F1 Score 1', 'AUC', 'AUC Probs 0', 'AUC Probs 1'];
		self.name_algorithms=['SGD', 'KNN', 'SVC', 'RF', 'DT'];

	def process_all_scores(self, run_imbalances):
		# for xval in range(0, len(run_imbalances.xval_sgd)):
		self.process_scores(run_imbalances.xval_sgd,run_imbalances.xval_knn, run_imbalances.xval_svc,run_imbalances.xval_rf,run_imbalances.xval_dt )


	def process_scores(self, xvalidation_sgd, xvalidation_knn, xvalidation_svc, xvalidation_rf, xvalidation_dt):
		xvalidations = [xvalidation_sgd, xvalidation_knn, xvalidation_svc, xvalidation_rf, xvalidation_dt];
		
		for xval_alg in range(0, len(xvalidations)):
			for xvalImb in range(0, len(xvalidations[xval_alg])):
				for ds_name in self.name_datasets:
					for measure in range(0, len(self.name_performances)):
						self.scores[xval_alg][measure][ds_name][xvalImb] = xvalidations[xval_alg][xvalImb].scores[ds_name][measure]


	def initialise_score_dics(self):
		accuracies = {'original':[0,0,0,0,0,0,0,0,0,0,0], 'oversample':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble_original':[0,0,0,0,0,0,0,0,0,0,0], 'smote':[0,0,0,0,0,0,0,0,0,0,0]}; #0
		precision0 = {'original':[0,0,0,0,0,0,0,0,0,0,0], 'oversample':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble_original':[0,0,0,0,0,0,0,0,0,0,0], 'smote':[0,0,0,0,0,0,0,0,0,0,0]}; #1
		precision1 = {'original':[0,0,0,0,0,0,0,0,0,0,0], 'oversample':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble_original':[0,0,0,0,0,0,0,0,0,0,0], 'smote':[0,0,0,0,0,0,0,0,0,0,0]}; #2
		recall0 = {'original':[0,0,0,0,0,0,0,0,0,0,0], 'oversample':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble_original':[0,0,0,0,0,0,0,0,0,0,0], 'smote':[0,0,0,0,0,0,0,0,0,0,0]}; #3
		recall1 = {'original':[0,0,0,0,0,0,0,0,0,0,0], 'oversample':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble_original':[0,0,0,0,0,0,0,0,0,0,0], 'smote':[0,0,0,0,0,0,0,0,0,0,0]}; #4
		balanced_accuracy = {'original':[0,0,0,0,0,0,0,0,0,0,0], 'oversample':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble_original':[0,0,0,0,0,0,0,0,0,0,0], 'smote':[0,0,0,0,0,0,0,0,0,0,0]}; #5
		f1score0 = {'original':[0,0,0,0,0,0,0,0,0,0,0], 'oversample':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble_original':[0,0,0,0,0,0,0,0,0,0,0], 'smote':[0,0,0,0,0,0,0,0,0,0,0]}; #6 
		f1score1 = {'original':[0,0,0,0,0,0,0,0,0,0,0], 'oversample':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble_original':[0,0,0,0,0,0,0,0,0,0,0], 'smote':[0,0,0,0,0,0,0,0,0,0,0]}; #7
		auc = {'original':[0,0,0,0,0,0,0,0,0,0,0], 'oversample':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble_original':[0,0,0,0,0,0,0,0,0,0,0], 'smote':[0,0,0,0,0,0,0,0,0,0,0]}; #8
		auc0 = {'original':[0,0,0,0,0,0,0,0,0,0,0], 'oversample':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble_original':[0,0,0,0,0,0,0,0,0,0,0], 'smote':[0,0,0,0,0,0,0,0,0,0,0]}; #9
		auc1 = {'original':[0,0,0,0,0,0,0,0,0,0,0], 'oversample':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble':[0,0,0,0,0,0,0,0,0,0,0], 'oversample_bubble_original':[0,0,0,0,0,0,0,0,0,0,0], 'smote':[0,0,0,0,0,0,0,0,0,0,0]}; #10
		scores=[accuracies, precision0, precision1, recall0, recall1, balanced_accuracy,f1score0, f1score1, auc, auc0, auc1];
		self.scores_sgd = deepcopy(scores);	
		self.scores_knn = deepcopy(scores);
		self.scores_svc = deepcopy(scores);
		self.scores_rf = deepcopy(scores);
		self.scores_dt = deepcopy(scores);
		self.scores=[self.scores_sgd, self.scores_knn, self.scores_svc, self.scores_rf, self.scores_dt]

	def run_all_performances(self, title):
		for i in range(0, len(self.name_performances)):
			self.plot_performances(i, title);

	def run_all_algorithms(self, title):
		for i in range(0, len(self.name_algorithms)):
			self.plot_algorithm(i, title)

	def plot_performances(self, performance, title):
		for i in range(0, len(self.name_algorithms)):
			title='Total_'+self.imbalance_name+'_'+self.name_performances[performance]+self.name_algorithms[i];
			self.plot_scores(i, performance, title)

	def plot_algorithm(self, algorithm, title):
		for i in range(0, len(self.name_performances)):
			title='Total_algorithm_'+self.imbalance_name+'_'+self.name_algorithms[algorithm]+'_'+self.name_performances[i];
			self.plot_scores(algorithm, i, title)

	def plot_scores(self, algorithm, performance, title):
		
		original=self.scores[algorithm][performance]['original'];
		oversample=self.scores[algorithm][performance]['oversample'];
		bubbleos=self.scores[algorithm][performance]['oversample_bubble'];
		bubbleoriginalos=self.scores[algorithm][performance]['oversample_bubble_original'];
		smote=self.scores[algorithm][performance]['smote'];

		y = range(0, len(original))

		titelfig = '../Figures/Performance_Figures/'+title+'.png'
		print(title)
		fig, host = plt.subplots(figsize=(20, 10));
		fig.subplots_adjust(right=0.75);
		par1 = host.twinx();
		par2 = host.twinx();
		par3 = host.twinx();
		par4 = host.twinx();
		par5 = host.twinx();       

		par2.spines["right"].set_position(("axes", 1.25))
		par3.spines["right"].set_position(("axes", 1.40))
		par4.spines["right"].set_position(("axes", 1.55))
		par5.spines["right"].set_position(("axes", 1.70))

		host.set_ylabel("Original");
		par1.set_ylabel("Random Oversampling");
		par2.set_ylabel("Original + Oversampling on Bubble");
		par3.set_ylabel("Oversampling on Bubble + Original");
		par4.set_ylabel("Smote Oversampling");     

		p1, = host.plot(y , original, color=('#b5e61d'), label="Original")
		p2, = par1.plot(y , oversample, color=("#ee15b8"), label="Random Oversampling")
		p3, = par2.plot(y , bubbleos, color=("#f86a0c"), label="Original + Oversampling on Bubble")
		p4, = par3.plot(y , bubbleoriginalos, color=("#3ac9a2"), label="Oversampling on Bubble + Original")  
		p5, = par4.plot(y , smote, color=("#ff0000"), label="Smote Oversampling")        

		lines = [p1, p2, p3, p4, p5]
		host.legend(lines, [l.get_label() for l in lines])

		host.yaxis.label.set_color(p1.get_color())
		par1.yaxis.label.set_color(p2.get_color())
		par2.yaxis.label.set_color(p3.get_color())
		par3.yaxis.label.set_color(p4.get_color())
		par4.yaxis.label.set_color(p5.get_color())
		 

		labels = ['50-50', '55-45', '60-40', '65-35', '70-30', '75-25', '80-20', '85-15','90-10','95-05'];
		host.set_xticklabels(labels)
		plt.savefig(titelfig, format='png');
		plt.show();



        
       