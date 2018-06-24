import pandas as pd;
import numpy as np;
import itertools;
from imblearn.over_sampling import RandomOverSampler as ros;
import copy;
from imblearn.under_sampling import RandomUnderSampler as rus;
from imblearn.over_sampling import SMOTE as smote;
import collections #debugging
import datetime;
import pickle;
from sklearn.model_selection import train_test_split


class Dataset:

	#Try conventions: minority class=1, majority class=0
	def __init__(self, data, X, y):
		self.data = data
		self.X = X;
		self.X_ordered=[];
		self.index_discrete=0  
		self.index_continuous=0;
		self.index_category=0;
		self.y = y;
		self.deltas = []; 
		self.deltas_ordered = [] #Smallest difference between two attributes for each feature
		self.entropy = []; #Entropy values for each of the features
		self.impurity = [];
		self.bubbles=pd.DataFrame(); #All the combination possible created from the mc with the bubble method
		self.number_of_originals = len(X); 
		self.bubble_feature_activation = []; #What combinations exist in order to create the dataset
		# self.nmOversample=nmOversample; #Number of minority instances
		#0-discrete, 1-continuous, 2-categorical
		self.categories = []; #Vector saying which columns are what type of attributes
		self.class_column=-1; #Column where the class is
		self.mc_label=1; #Label of the minority class
		self.oversampling_proportion = {}; #With what proportion should we oversample - 
		self.rn_state=0; #Random State to find the same oversampling datasets

		self.original_training_set=[]; #Normal split training set 
		self.original_test_set=[];#Normal split test set
		self.oversample_tt_set=[]; #Random oversampling of the minority class
		self.oversample_bubble_tt_set=[]; #Oversampling is only from the bubble set, take all normal instance
		self.oversample_bubble_original_tt_set=[] #Oversample overall with bubble methods
		self.all_bubbles_tt_set=[];
		self.smote_bubble_tt_set=[]; #Oversample overall dataset with smote (scikit method)

		self.prop_bubble={};
		self.prop_bubble_original={};



	#Find all combinations in which to input noise
	def create_featurevect(self):
		length = len(self.X.iloc[0]);
		liste = list(itertools.product([0,1], repeat=length))[1:];
		for i in range(0, len(liste)):
			liste[i] = np.array(liste[i]);
		self.bubble_feature_activation = np.array(liste)

	def set_props(self):
		maxim = max(self.original_training_set['ys'][0].value_counts());
		minim = min(self.original_training_set['ys'][0].value_counts());
		self.prop_bubble={0:maxim, 1:maxim-minim};
		self.prop_bubble_original={0:maxim, 1:maxim};


	#Find smallest differences (uses find_smaller_step)
	def compute_deltas(self):
		#category: 0-discrete, 1-continuous, 2-categorical
		deltas = [];
		for i in range(0, len(self.categories)):
			if self.categories[i]==0 or self.categories[i]==2:
				deltas.append(self.find_smaller_step(self.X.iloc[:][i]));
			elif self.categories[i]==1:
				deltas.append(self.find_smaller_step(self.X.iloc[:][i]));
		self.deltas = np.array(deltas);

	def find_smaller_step(self,attributes):
		difference = 100;
		att = attributes.sort_values();
		for i in range(0, (len(att)-1)):
			diff = att.iloc[i+1] - att.iloc[i];
			if (diff != 0):
				if (diff < difference):
					difference = diff;
		return difference

	def setClass_column(self, mc_column):
		self.class_column=class_column;

	def setMc_label(self, mc_label):
		self.mc_label=mc_label;

	def setOversampling_proportion(self, oversampling_proportion):
		self.oversampling_proportion = oversampling_proportion;

	#Order the features of the data to have new datasets = col(discrete)|col(continuous)|col(category)|y
	def setCategories(self,categories):
		print('setting')
		print('props', self.prop_bubble);
		self.categories = np.array(categories);
		self.compute_deltas();
		idiscrete = [index for index, value in enumerate(self.categories) if value==0];
		icontinuous = [index for index, value in enumerate(self.categories) if value==1];
		icategory = [index for index, value in enumerate(self.categories) if value==2];
		discrete=self.X[idiscrete]
		continuous=self.X[icontinuous]
		category=self.X[icategory]
		self.X_ordered=pd.concat([discrete, continuous,category,self.y], axis=1)
		self.index_discrete = len(idiscrete);
		self.index_continuous = len(icontinuous)+self.index_discrete;
		self.index_category=len(icategory)+self.index_continuous;
		delta_discrete=pd.DataFrame(self.deltas[idiscrete]);
		delta_continuous=pd.DataFrame(self.deltas[icontinuous]);
		delta_category=pd.DataFrame(self.deltas[icategory]);
		self.deltas_ordered=np.concatenate([delta_discrete, delta_continuous,delta_category], axis=0).flatten();

	def setDeltas(self,deltas):
		self.deltas=deltas;
	def setEntropy(self,entropy):
		self.entropy=entropy;
	def setImpurity(self,impurity):
		self.impurity=impurity;
	def setBubbles(self,bubbles):
		self.bubbles = bubbles;
	def setOversamplePool(self,oversample_pool):
		self.oversample_pool = oversample_pool;		
	def setRnstate(self, rn_state):
		self.rn_state=rn_state;


	#Create neutral validation test to test all of of the algorithms
	def tt_split(self):
		random_states = [9,97,42,21,7,36,5]; #In order to reproduce the experiment. Randomly chosen: 9-favourite number, 97-date of birth, 42-answer of life, 21-fun card game, 7-lucky number, 36-cool number cause divisible by so many numbers, 5-day of birth
		xTrain=[]
		xTest=[]
		yTrain=[]
		yTest=[]

		currx = self.X_ordered.iloc[:, :-1];
		curry = self.X_ordered.iloc[:, -1]

		for state in random_states:
			x_Train, x_Test, y_Train, y_Test = train_test_split(currx, curry, test_size=0.3, random_state=state, stratify=curry);
			xTrain.append(x_Train);xTest.append(x_Test);yTrain.append(y_Train);yTest.append(y_Test);

		self.original_training_set = {'xs':xTrain,'ys':yTrain}
		self.original_test_set = {'xs':xTest, 'ys':yTest} 



	def bulle(self, trainX, trainY):
		#categories: 0-discrete, 1-continuous, 2-category
		ordered=pd.concat([trainX, trainY], axis=1)
		x=pd.DataFrame(trainX[trainY==self.mc_label])
		i=0;
		t00=datetime.datetime.now().replace(microsecond=0);
		# print("Starting!!",t0);
		bubble_set=pd.DataFrame();
		# combination=list(np.ones(len(self.categories)).astype(np.int));
		for combination in range(0, len(self.bubble_feature_activation)):
			# if i%500==0:
			# 	tnoow=datetime.datetime.now().replace(microsecond=0);
			# 	print("aan het bullen", i, tnoow-t00);	 
			# 	t00=tnoow;	
			step_matrix = np.random.rand(len(x), len(self.deltas_ordered))*self.deltas_ordered;
			discrete = x.iloc[:, 0:self.index_discrete]+step_matrix[:, 0:self.index_discrete]-(self.deltas_ordered[0:self.index_discrete]/2);
			continuous = x.iloc[:,self.index_discrete:self.index_continuous]+(step_matrix[:,self.index_discrete:self.index_continuous]*2)-(self.deltas_ordered[self.index_discrete:self.index_continuous]);
			category = x.iloc[:,self.index_continuous:]+step_matrix[:,self.index_continuous:];
			bubble = pd.concat([discrete, continuous, category], axis=1)
			bubble_set=bubble_set.append(bubble)			
			i+=1

		bubble_y = np.ones(len(self.bubble_feature_activation)*len(x)).astype(np.int_)
		return bubble_set, pd.DataFrame(bubble_y);

	#Random Oversampling on the data
	def random_oversample(self):
		xTrainOversampled=[];
		yTrainOversampled=[];
		print("Random Oversample")
		for fold in range(0, len(self.original_training_set['xs'])):
			maxim = max(self.original_training_set['ys'][fold].value_counts());
			proportion = {0:maxim, 1:maxim}
			randomOversampler = ros(random_state=self.rn_state, ratio = proportion);
			Xoversampling, Yoversampling = randomOversampler.fit_sample(self.original_training_set['xs'][fold], self.original_training_set['ys'][fold].values.ravel());
			xTrainOversampled.append(Xoversampling);
			yTrainOversampled.append(Yoversampling);
		self.oversample_tt_set={'xs':xTrainOversampled, 'ys':yTrainOversampled};


	def bubble_oversample(self):
		xBubbleOversample=[];
		yBubbleOversample=[];

		xBubbleOriginalOversample=[];
		yBubbleOriginalOversample=[];

		xAllOversample=[];
		yAllOversample=[];

		print("Bubble Oversample");
		t00=datetime.datetime.now().replace(microsecond=0)
		for fold in range(0, len(self.original_training_set['xs'])):
			tnoow=datetime.datetime.now().replace(microsecond=0);
			print("aan het bullen",fold, tnoow-t00);
			t00=tnoow;	

			maxim = max(self.original_training_set['ys'][fold].value_counts());
			minim = min(self.original_training_set['ys'][fold].value_counts());
			prop_bubble={0:maxim, 1:maxim-minim};
			prop_bubble_original={0:maxim, 1:maxim};
			# self.prop_bubble=prop_bubble;
			# self.prop_bubble_original=prop_bubble_original;

			bubblex, bubbley = self.bulle(self.original_training_set['xs'][fold], self.original_training_set['ys'][fold])
			majcX=pd.DataFrame(self.original_training_set['xs'][fold][(self.original_training_set['ys'][fold]==0)])
			majcy=pd.DataFrame(self.original_training_set['ys'][fold][(self.original_training_set['ys'][fold]==0)])
			mincX=pd.DataFrame(self.original_training_set['xs'][fold][(self.original_training_set['ys'][fold]==1)])
			mincy=pd.DataFrame(self.original_training_set['ys'][fold][(self.original_training_set['ys'][fold]==1)])

			majcX = majcX.append(bubblex, ignore_index=True);
			majcy.columns=bubbley.columns
			majcy = pd.concat([majcy,bubbley], ignore_index=True, axis=0);
         

			randomUnderSampler = rus(random_state=self.rn_state, ratio=self.prop_bubble); 
			x_bubbleO, y_bubbleO = randomUnderSampler.fit_sample(majcX, majcy.values.ravel());
			x_bubbleO=pd.DataFrame(x_bubbleO).append(mincX, ignore_index=0);
			y_bubbleO=pd.DataFrame(y_bubbleO).append(mincy, ignore_index=0);

			xBubbleOversample.append(x_bubbleO)
			yBubbleOversample.append(y_bubbleO)		


			randomUnderSampler = rus(random_state=self.rn_state, ratio=self.prop_bubble_original);
			oribubblex = self.original_training_set['xs'][fold].append(bubblex);
			oribubbley = self.original_training_set['ys'][fold].append(bubbley);
			x_bubbleOriO, y_bubbleOriO = randomUnderSampler.fit_sample(oribubblex, oribubbley.values.ravel());

			xBubbleOriginalOversample.append(x_bubbleOriO)
			yBubbleOriginalOversample.append(y_bubbleOriO)

			xAllOversample.append(np.array(oribubblex));
			yAllOversample.append(np.array(oribubbley.values.ravel()));

			

		self.oversample_bubble_tt_set = {'xs':xBubbleOversample, 'ys':yBubbleOversample};
		self.oversample_bubble_original_tt_set = {'xs':xBubbleOriginalOversample, 'ys':yBubbleOriginalOversample};
		self.all_bubbles_tt_set={'xs':xAllOversample, 'ys':yAllOversample};


	def smote_oversample(self):
		xSmote=[];
		ySmote=[];
		print("Smote Oversample")
		rn_states = [29,9,5,10,8,26,4]
		
		for fold in range(0, len(self.original_training_set['xs'])):
			maxim = max(self.original_training_set['ys'][fold].value_counts());
			oversampling_proportion={0:maxim, 1:maxim}
			smoteOversampler = smote(random_state=rn_states[fold], ratio=oversampling_proportion);
			Xoversampling, Yoversampling = smoteOversampler.fit_sample(self.original_training_set['xs'][fold], self.original_training_set['ys'][fold].values.ravel());
			xSmote.append(Xoversampling)
			ySmote.append(Yoversampling)

		self.smote_bubble_tt_set = {'xs':xSmote, 'ys':ySmote};

	def letsBulle(self):
		t0=datetime.datetime.now().replace(microsecond=0);
		print("FEATURE VECTOR");
		self.create_featurevect();
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('Time taken for creating the feature vectors: ', tnow-t0)
		t0=tnow;

		print('ORDER THE DATA');
		self.setCategories(self.categories);
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('Time taken for ordering the data: ', tnow-t0)
		t0=tnow;

		print("SPLIT")
		self.tt_split();
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('Time taken for split: ', tnow-t0)
		t0=tnow;

		self.set_props();

		print('RANDOM OVERSAMPLING')
		self.random_oversample();		
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('Time taken for random oversampling: ', tnow-t0)
		t0=tnow;

		print('BUBBLE OVERSAMPLE')
		self.bubble_oversample();
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('Time taken for random bubble oversampling: ', tnow-t0)
		t0=tnow;

		print('SMOTE OVERSAMPLE')
		self.smote_oversample();
		tnow=datetime.datetime.now().replace(microsecond=0);
		print('Time taken for random SMOTE: ', tnow-t0)
		t0=tnow;



			
		



