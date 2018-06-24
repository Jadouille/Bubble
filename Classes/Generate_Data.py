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
import BuildMyBulle;
from sklearn import datasets

class Generate_Data:
	def __init__(self, datasetLabel, bulle, bubble_oversample, categories, name_experiment):
		self.datasetLabel=datasetLabel;
		self.DummyDataset=[];
		self.datasetNames=['haberman', 'pulsar', 'magic', 'tictactoe', 'Maxi', 'Midi', 'Mini', 'Tiny', 'Micro', 'Nano']
		self.categories = categories;
		self.bulle=bulle;
		self.bubble_oversample=bubble_oversample
		self.name_experiment = name_experiment;
		print('youp')

	def scale(self,attributes):
		scaler = StandardScaler();
		return pd.DataFrame(scaler.fit_transform(attributes)), scaler;

	def save_xvalidation(self, obj, path):
		with open(path, 'wb') as fp:
			pickle.dump(object, fp)



	def instantiate_haberman(self):
		with open('../DataSets/Haberman.txt', 'rb') as fp:
			haberman=pickle.load(fp);
		#Transform class 1 and 2 into 0 and 1
		for instance in range(0, len(haberman)):
			haberman.iloc[instance][3] -= 1;

		#Separte class and attributes
		hab_attributes = pd.DataFrame(haberman[[0,1,2]]);
		hab_class = haberman[[3]];

		#Scale the original data
		hab_attributes, scaler = self.scale(hab_attributes);
		haberman = hab_attributes.copy();
		haberman[3] = hab_class;
		mc_haberman = haberman[haberman[3]==1]

		#Creation of the Dataset
		DummyHaberman = Dataset_Double.Dataset(data=haberman, X=hab_attributes, y=hab_class)
		DummyHaberman.setCategories(self.categories);
		DummyHaberman.rn_state=9
		DummyHaberman.class_column=3;

		# print(DummyHaberman.prop_bubble_original)
		if self.bulle != 0:
			DummyHaberman.bulle= MethodType(self.bulle, DummyHaberman)
		if self.bubble_oversample !=0:
			DummyHaberman.bubble_oversample=MethodType(self.bubble_oversample, DummyHaberman)
		DummyHaberman.letsBulle()

		for y in range(0, len(DummyHaberman.oversample_bubble_tt_set['ys'])):
			DummyHaberman.oversample_bubble_tt_set['ys'][y]=DummyHaberman.oversample_bubble_tt_set['ys'][y].replace(np.nan, 0)
			DummyHaberman.oversample_bubble_tt_set['ys'][y]['class']=DummyHaberman.oversample_bubble_tt_set['ys'][y][0]+DummyHaberman.oversample_bubble_tt_set['ys'][y][3]
			DummyHaberman.oversample_bubble_tt_set['ys'][y] = np.array(DummyHaberman.oversample_bubble_tt_set['ys'][y]['class'])

		self.DummyDataset=DummyHaberman;

		path = '../DataSets/oversampled/experimentdouble_'+self.name_experiment+self.datasetNames[self.datasetLabel]+'.txt';
		with open(path, 'wb') as fp:
			pickle.dump(DummyHaberman,fp)


	def instantiate_pulsar(self):
		pulsar = pd.read_csv('../Datasets/HTRU_2.csv', header = None);
		#Separte class and attributes
		pulsar_attributes = pd.DataFrame(pulsar[[0,1,2,3,4,5,6,7]]);
		pulsar_class = pulsar[[8]];

		#Scale the original data
		pulsar_attributes, scaler = self.scale(pulsar_attributes);
		pulsar = pulsar_attributes.copy();
		pulsar[8] = pulsar_class;
		mc_pulsar = pulsar[pulsar[8]==1]

		#Creation of the Dataset
		DummyPulsar = Dataset_Double.Dataset(data=pulsar, X=pulsar_attributes, y=pulsar_class)
		DummyPulsar.setCategories(self.categories)
		DummyPulsar.rn_state=9
		DummyPulsar.class_column=8;
		if self.bulle != 0:	
			DummyPulsar.bulle= MethodType(self.bulle, DummyPulsar)
		if self.bubble_oversample !=0:
			DummyPulsar.bubble_oversample=MethodType(self.bubble_oversample, DummyPulsar)

		DummyPulsar.letsBulle()

		for y in range(0, len(DummyPulsar.oversample_bubble_tt_set['ys'])):
			DummyPulsar.oversample_bubble_tt_set['ys'][y]=DummyPulsar.oversample_bubble_tt_set['ys'][y].replace(np.nan, 0)
			DummyPulsar.oversample_bubble_tt_set['ys'][y]['class']=DummyPulsar.oversample_bubble_tt_set['ys'][y][0]+DummyPulsar.oversample_bubble_tt_set['ys'][y][8]
			DummyPulsar.oversample_bubble_tt_set['ys'][y] = np.array(DummyPulsar.oversample_bubble_tt_set['ys'][y]['class'])

		self.DummyDataset=DummyPulsar;

		path = '../DataSets/oversampled/experimentdouble_'+self.name_experiment+self.datasetNames[self.datasetLabel]+'.txt';
		with open(path, 'wb') as fp:
			pickle.dump(DummyPulsar,fp)


	def instantiate_magic(self):
		magic = pd.read_csv('../DataSets/Magic.csv', header = None);
		#Separte class and attributes
		magic=magic.replace('g', 0);
		magic=magic.replace('h',1);

		magic_attributes = pd.DataFrame(magic[[0,1,2,3,4,5,6,7,8,9,10]]);
		magic_class = magic[[11]];

		#Scale the original data
		magic_attributes, scaler = self.scale(magic_attributes);
		magic = magic_attributes.copy();
		magic[11] = magic_class;
		mc_magic = magic[magic[11]==1]

		#Creation of the Dataset
		DummyMagic = Dataset_Double.Dataset(data=magic, X=magic_attributes, y=magic_class)
		DummyMagic.setCategories(self.categories)
		DummyMagic.rn_state=9
		DummyMagic.class_column=11;

		if self.bulle != 0:
			DummyMagic.bulle= MethodType(self.bulle, DummyMagic)
		if self.bubble_oversample !=0:
			DummyMagic.bubble_oversample=MethodType(self.bubble_oversample, DummyMagic)

		DummyMagic.letsBulle()

		for y in range(0, len(DummyMagic.oversample_bubble_tt_set['ys'])):
			DummyMagic.oversample_bubble_tt_set['ys'][y]=DummyMagic.oversample_bubble_tt_set['ys'][y].replace(np.nan, 0)
			DummyMagic.oversample_bubble_tt_set['ys'][y]['class']=DummyMagic.oversample_bubble_tt_set['ys'][y][0]+DummyMagic.oversample_bubble_tt_set['ys'][y][11]
			DummyMagic.oversample_bubble_tt_set['ys'][y] = np.array(DummyMagic.oversample_bubble_tt_set['ys'][y]['class'])
		self.DummyDataset=DummyMagic;

		path = '../DataSets/oversampled/experimentdouble_'+self.name_experiment+self.datasetNames[self.datasetLabel]+'.txt';
		with open(path, 'wb') as fp:
			pickle.dump(DummyMagic,fp)

	def instantiate_tictactoe(self):
		tictactoe = pd.read_csv('../DataSets/TicTacToe.csv', header = None, index_col=0);
		#Separte class and attributes
		tictactoe = tictactoe.replace('x', 1);
		tictactoe = tictactoe.replace('o', 2);
		tictactoe = tictactoe.replace('b', 3);
		tictactoe = tictactoe.replace('positive', 0)
		tictactoe = tictactoe.replace('negative', 1)
		tictactoe_attributes = pd.DataFrame(tictactoe[[1,2,3,4,5,6,7,8,9]]);
		tictactoe_class = tictactoe[[10]];

		#Scale the original data
		tictactoe_attributes, scaler = self.scale(tictactoe_attributes);
		tictactoe = tictactoe_attributes.copy();
		tictactoe[10] = tictactoe_class;
		mc_tictactoe = tictactoe[tictactoe[10]==1]

		#Creation of the Dataset
		DummyTictactoe = Dataset_Double.Dataset(data=tictactoe, X=tictactoe_attributes, y=tictactoe_class)
		DummyTictactoe.setCategories(self.categories);
		DummyTictactoe.rn_state=9;
		DummyTictactoe.class_column=10;

		if self.bulle != 0:
			DummyTictactoe.bulle= MethodType(self.bulle, DummyTictactoe)
		if self.bubble_oversample !=0:
			DummyTictactoe.bubble_oversample=MethodType(self.bubble_oversample, DummyTictactoe)

		DummyTictactoe.letsBulle()

		for y in range(0, len(DummyTictactoe.oversample_bubble_tt_set['ys'])):
			DummyTictactoe.oversample_bubble_tt_set['ys'][y]=DummyTictactoe.oversample_bubble_tt_set['ys'][y].replace(np.nan, 0)
			DummyTictactoe.oversample_bubble_tt_set['ys'][y]['class']=DummyTictactoe.oversample_bubble_tt_set['ys'][y][0]+DummyTictactoe.oversample_bubble_tt_set['ys'][y][10]
			DummyTictactoe.oversample_bubble_tt_set['ys'][y] = np.array(DummyTictactoe.oversample_bubble_tt_set['ys'][y]['class'])
		self.DummyDataset=DummyTictactoe;

		path = '../DataSets/oversampled/experimentdouble_'+self.name_experiment+self.datasetNames[self.datasetLabel]+'.txt';
		with open(path, 'wb') as fp:
			pickle.dump(DummyTictactoe,fp)



	def instantiate_imbalances(self):
		path='../DataSets/Unbalanced/'
		n_samples=0;
		if self.datasetLabel==4:
			n_samples=10000;
			name='Maxi';
		elif self.datasetLabel==5:
			n_samples=5000;
			name="Midi";
		elif self.datasetLabel==6:
			n_samples=2000;
			name="Mini";
		elif self.datasetLabel==7:
			n_samples=500;
			name="Tiny";

		elif self.datasetLabel==8:
			n_samples=300;
			name="Micro";

		elif self.datasetLabel==9:
			n_samples=150;
			name="Nano";


		#Minority Class = 50%
		fiftyX, fiftyY = datasets.make_classification(n_samples=n_samples, n_features=7, n_informative=4, n_redundant=3, random_state=42, weights=[1,0.99])
		#Separte class and attributes
		fifty_attributes = pd.DataFrame(fiftyX);
		fifty_class = pd.DataFrame(fiftyY);
		#Scale the original data
		fifty_attributes, scaler = self.scale(fifty_attributes);
		fifty = fifty_attributes.copy();
		fifty[12] = fifty_class;
		mc_fifty = fifty[fifty[12]==1]
		#Creation of the Dataset
		DummyFifty = Dataset_Double.Dataset(data=fifty, X=fifty_attributes, y=fifty_class)
		DummyFifty.setCategories([1,1,1,1,1,1,1])
		DummyFifty.rn_state=9
		DummyFifty.class_column=12;
		if self.bulle != 0:
			DummyFifty.bulle= MethodType(self.bulle, DummyFifty)
		if self.bubble_oversample !=0:
			DummyFifty.bubble_oversample=MethodType(self.bubble_oversample, DummyFifty)
		DummyFifty.letsBulle()

		temppath=path+self.name_experiment+name+'_Fifty.txt'
		with open(temppath, 'wb') as fp:
			pickle.dump(DummyFifty,fp,pickle.HIGHEST_PROTOCOL)


		#Minority Class = 55%
		fifty5X, fifty5Y = datasets.make_classification(n_samples=n_samples, n_features=7, n_informative=4, n_redundant=3, random_state=42, weights=[1,0.9])
		#Separte class and attributes
		fifty5_attributes = pd.DataFrame(fifty5X);
		fifty5_class = pd.DataFrame(fifty5Y);
		#Scale the original data
		fifty5_attributes, scaler = self.scale(fifty5_attributes);
		fifty5 = fifty5_attributes.copy();
		fifty5[12] = fifty5_class;
		mc_fifty5 = fifty5[fifty5[12]==1]
		#Creation of the Dataset
		DummyFifty5 = Dataset_Double.Dataset(data=fifty5, X=fifty5_attributes, y=fifty5_class)
		DummyFifty5.setCategories([1,1,1,1,1,1,1])
		DummyFifty5.rn_state=9
		DummyFifty5.class_column=12;
		if self.bulle != 0:
			DummyFifty5.bulle= MethodType(self.bulle, DummyFifty5)
		if self.bubble_oversample !=0:
			DummyFifty5.bubble_oversample=MethodType(self.bubble_oversample, DummyFifty5)
		DummyFifty5.letsBulle()
		temppath=path+self.name_experiment+name+'_Fifty5.txt'
		with open(temppath, 'wb') as fp:
		    pickle.dump(DummyFifty5,fp,pickle.HIGHEST_PROTOCOL)


		#Minority Class = 60%
		sixtyX, sixtyY = datasets.make_classification(n_samples=n_samples, n_features=7, n_informative=4, n_redundant=3, random_state=42, weights=[1,0.8])
		#Separte class and attributes
		sixty_attributes = pd.DataFrame(sixtyX);
		sixty_class = pd.DataFrame(sixtyY);
		#Scale the original data
		sixty_attributes, scaler = self.scale(sixty_attributes);
		sixty = sixty_attributes.copy();
		sixty[12] = sixty_class;
		mc_sixty = sixty[sixty[12]==1]
		#Creation of the Dataset
		DummySixty = Dataset_Double.Dataset(data=sixty, X=sixty_attributes, y=sixty_class)
		DummySixty.setCategories([1,1,1,1,1,1,1])
		DummySixty.rn_state=9
		DummySixty.class_column=12;
		if self.bulle != 0:
			DummySixty.bulle= MethodType(self.bulle, DummySixty)
		if self.bubble_oversample !=0:
			DummySixty.bubble_oversample=MethodType(self.bubble_oversample, DummySixty)
		DummySixty.letsBulle()
		temppath=path+self.name_experiment+name+'_Sixty.txt'
		with open(temppath, 'wb') as fp:
			pickle.dump(DummySixty,fp,pickle.HIGHEST_PROTOCOL)


		#Minority Class = 65%
		sixty5X, sixty5Y = datasets.make_classification(n_samples=n_samples, n_features=7, n_informative=4, n_redundant=3, random_state=42, weights=[1,0.7])
		#Separte class and attributes
		sixty5_attributes = pd.DataFrame(sixty5X);
		sixty5_class = pd.DataFrame(sixty5Y);
		#Scale the original data
		sixty5_attributes, scaler = self.scale(sixty5_attributes);
		sixty5 = sixty5_attributes.copy();
		sixty5[12] = sixty5_class;
		mc_sixty5 = sixty5[sixty5[12]==1]
		#Creation of the Dataset
		DummySixty5 = Dataset_Double.Dataset(data=sixty5, X=sixty5_attributes, y=sixty5_class)
		DummySixty5.setCategories([1,1,1,1,1,1,1])
		DummySixty5.rn_state=9
		DummySixty5.class_column=12;
		if self.bulle != 0:
			DummySixty5.bulle= MethodType(self.bulle, DummySixty5)
		if self.bubble_oversample !=0:
			DummySixty5.bubble_oversample=MethodType(self.bubble_oversample, DummySixty5)
		DummySixty5.letsBulle()
		temppath=path+self.name_experiment+name+'_Sixty5.txt'
		with open(temppath, 'wb') as fp:
		    pickle.dump(DummySixty5,fp,pickle.HIGHEST_PROTOCOL)

		#Minority Class = 70%
		seventyX, seventyY = datasets.make_classification(n_samples=n_samples, n_features=7, n_informative=4, n_redundant=3, random_state=42, weights=[1,0.6])
		#Separte class and attributes
		seventy_attributes = pd.DataFrame(seventyX);
		seventy_class = pd.DataFrame(seventyY);
		#Scale the original data
		seventy_attributes, scaler = self.scale(seventy_attributes);
		seventy = seventy_attributes.copy();
		seventy[12] = seventy_class;
		mc_seventy = seventy[seventy[12]==1]
		#Creation of the Dataset
		DummySeventy = Dataset_Double.Dataset(data=seventy, X=seventy_attributes, y=seventy_class)
		DummySeventy.setCategories([1,1,1,1,1,1,1])
		DummySeventy.rn_state=9
		DummySeventy.class_column=12;
		if self.bulle != 0:
			DummySeventy.bulle= MethodType(self.bulle, DummySeventy)
		if self.bubble_oversample !=0:
			DummySeventy.bubble_oversample=MethodType(self.bubble_oversample, DummySeventy)
		DummySeventy.letsBulle()
		temppath=path+self.name_experiment+name+'_Seventy.txt'
		with open(temppath, 'wb') as fp:
			pickle.dump(DummySeventy,fp,pickle.HIGHEST_PROTOCOL)

		#Minority Class = 75%
		seventy5X, seventy5Y = datasets.make_classification(n_samples=n_samples, n_features=7, n_informative=4, n_redundant=3, random_state=42, weights=[1,0.5])
		#Separte class and attributes
		seventy5_attributes = pd.DataFrame(seventy5X);
		seventy5_class = pd.DataFrame(seventy5Y);
		#Scale the original data
		seventy5_attributes, scaler = self.scale(seventy5_attributes);
		seventy5 = seventy5_attributes.copy();
		seventy5[12] = seventy5_class;
		mc_seventy5 = seventy5[seventy5[12]==1]
		#Creation of the Dataset
		DummySeventy5 = Dataset_Double.Dataset(data=seventy5, X=seventy5_attributes, y=seventy5_class)
		DummySeventy5.setCategories([1,1,1,1,1,1,1])
		DummySeventy5.rn_state=9
		DummySeventy5.class_column=12;
		if self.bulle != 0:
			DummySeventy5.bulle= MethodType(self.bulle, DummySeventy5)
		if self.bubble_oversample !=0:
			DummySeventy5.bubble_oversample=MethodType(self.bubble_oversample, DummySeventy5)
		DummySeventy5.letsBulle()
		temppath=path+self.name_experiment+name+'_Seventy5.txt'
		with open(temppath, 'wb') as fp:
			pickle.dump(DummySeventy5,fp,pickle.HIGHEST_PROTOCOL)


		#Minority Class = 80%
		eightyX, eightyY = datasets.make_classification(n_samples=n_samples, n_features=7, n_informative=4, n_redundant=3, random_state=42, weights=[1,0.4])
		#Separte class and attributes
		eighty_attributes = pd.DataFrame(eightyX);
		eighty_class = pd.DataFrame(eightyY);
		#Scale the original data
		eighty_attributes, scaler = self.scale(eighty_attributes);
		eighty = eighty_attributes.copy();
		eighty[12] = eighty_class;
		mc_eighty = eighty[eighty[12]==1]
		#Creation of the Dataset
		DummyEighty = Dataset_Double.Dataset(data=eighty, X=eighty_attributes, y=eighty_class)
		DummyEighty.setCategories([1,1,1,1,1,1,1])
		DummyEighty.rn_state=9
		DummyEighty.class_column=12;
		if self.bulle != 0:
			DummyEighty.bulle= MethodType(self.bulle, DummyEighty)
		if self.bubble_oversample !=0:
			DummyEighty.bubble_oversample=MethodType(self.bubble_oversample, DummyEighty)
		DummyEighty.letsBulle()
		temppath=path+self.name_experiment+name+'_Eighty.txt'
		with open(temppath, 'wb') as fp:
			pickle.dump(DummyEighty,fp,pickle.HIGHEST_PROTOCOL)

		#Minority Class = 85%
		eighty5X, eighty5Y = datasets.make_classification(n_samples=n_samples, n_features=7, n_informative=4, n_redundant=3, random_state=42, weights=[1,0.3])
		#Separte class and attributes
		eighty5_attributes = pd.DataFrame(eighty5X);
		eighty5_class = pd.DataFrame(eighty5Y);
		#Scale the original data
		eighty5_attributes, scaler = self.scale(eighty5_attributes);
		eighty5 = eighty5_attributes.copy();
		eighty5[12] = eighty5_class;
		mc_eighty5 = eighty5[eighty5[12]==1]
		#Creation of the Dataset
		DummyEighty5 = Dataset_Double.Dataset(data=eighty5, X=eighty5_attributes, y=eighty5_class)
		DummyEighty5.setCategories([1,1,1,1,1,1,1])
		DummyEighty5.rn_state=9
		DummyEighty5.class_column=12;
		if self.bulle != 0:
			DummyEighty5.bulle= MethodType(self.bulle, DummyEighty5)
		if self.bubble_oversample !=0:
			DummyEighty5.bubble_oversample=MethodType(self.bubble_oversample, DummyEighty5)
		DummyEighty5.letsBulle()
		temppath=path+self.name_experiment+name+'_Eighty5.txt'
		with open(temppath, 'wb') as fp:
		    pickle.dump(DummyEighty5,fp,pickle.HIGHEST_PROTOCOL)

		#Minority Class = 90%
		ninetyX, ninetyY = datasets.make_classification(n_samples=n_samples, n_features=7, n_informative=4, n_redundant=3, random_state=42, weights=[1,0.2])
		#Separte class and attributes
		ninety_attributes = pd.DataFrame(ninetyX);
		ninety_class = pd.DataFrame(ninetyY);
		#Scale the original data
		ninety_attributes, scaler = self.scale(ninety_attributes);
		ninety = ninety_attributes.copy();
		ninety[12] = ninety_class;
		mc_ninety = ninety[ninety[12]==1]
		#Creation of the Dataset
		DummyNinety = Dataset_Double.Dataset(data=ninety, X=ninety_attributes, y=ninety_class)
		DummyNinety.setCategories([1,1,1,1,1,1,1])
		DummyNinety.rn_state=9
		DummyNinety.class_column=12;
		if self.bulle != 0:
			DummyNinety.bulle= MethodType(self.bulle, DummyNinety)
		if self.bubble_oversample !=0:
			DummyNinety.bubble_oversample=MethodType(self.bubble_oversample, DummyNinety)
		DummyNinety.letsBulle()
		temppath=path+self.name_experiment+name+'_Ninety.txt'
		with open(temppath, 'wb') as fp:
		    pickle.dump(DummyNinety,fp,pickle.HIGHEST_PROTOCOL)

		#Minority Class = 95%
		ninety5X, ninety5Y = datasets.make_classification(n_samples=n_samples, n_features=7, n_informative=4, n_redundant=3, random_state=42, weights=[1,0.1])
		#Separte class and attributes
		ninety5_attributes = pd.DataFrame(ninety5X);
		ninety5_class = pd.DataFrame(ninety5Y);
		#Scale the original data
		ninety5_attributes, scaler = self.scale(ninety5_attributes);
		ninety5 = ninety5_attributes.copy();
		ninety5[12] = ninety5_class;
		mc_ninety5 = ninety5[ninety5[12]==1]
		#Creation of the Dataset
		DummyNinety5 = Dataset_Double.Dataset(data=ninety5, X=ninety5_attributes, y=ninety5_class)
		DummyNinety5.setCategories([1,1,1,1,1,1,1])
		DummyNinety5.rn_state=9
		DummyNinety5.class_column=12;
		if self.bulle != 0:
			DummyNinety.bulle= MethodType(self.bulle, DummyNinety)
		if self.bubble_oversample !=0:
			DummyNinety.bubble_oversample=MethodType(self.bubble_oversample, DummyNinety)
		DummyNinety5.letsBulle()
		temppath=path+self.name_experiment+name+'_Ninety5.txt'
		with open(temppath, 'wb') as fp:
			pickle.dump(DummyNinety5,fp,pickle.HIGHEST_PROTOCOL)














