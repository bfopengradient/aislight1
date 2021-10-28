import numpy as np
import pandas as pd
import streamlit as st

 
"""app allows user to add or delete trainig data"""


def app():
	st.write('#')
	st.markdown(' ### Edit or replace training data')
	st.write('#')
	

	#Load training data
	def load_training():
		train= pd.read_csv('./apps/train_ais.csv',usecols=['sentences','labels'])
		return train
 
 
	#Update with new sentence and label function 
	def enter_sentence_category(): 
		new_data=[] 
		new={}    
		#need two text fields

		new['sentences']=st.text_input("Enter sentence for training data set: either something of interest or not")
		st.write(new['sentences'])
		#new['sentences']= new_sent# .strip()
		new['labels']=st.number_input('Enter label: 1 for sentence of interest and 0 for sentence not of interest', min_value=0, max_value=1)
		new_data.append(new) 
		return new_data

	#Add individual sentence and label 	
	def new_sentence():
		add_sentence= st.checkbox('Add a new sentence and label to the training dataset')
		if add_sentence:
		#create new data frame, align columns with input data and write in append mode to csv
			train_1= pd.DataFrame(columns={'labels','sentences'})
			train_1=train_1.loc[:,['sentences','labels']]
			#call update function and append results to test csv
			train_1= train_1.append(enter_sentence_category(),ignore_index=True) 
			st.write('#')
			add_data= st.button("Update training data ")
			if add_data:
				st.write(":thumbsup:")
				return train_1.to_csv('./apps/train_ais.csv', mode='a', header=False,index=False)

	#Add/upload csv  file 
	def replace_file():	 
		add_file=st.checkbox('Replace current training data csv file with new one ')
		if add_file:
			st.write('Uploaded file must have just two columns: sentences and labels ')			 
			dataset = st.file_uploader("Upload file here", type = ['csv'])
			if dataset is not None:
				train_2 = pd.read_csv(dataset)
				st.write('Displying top of the uploaded csv file')
				st.dataframe(train_2.head(5))
				st.write(":thumbsup:")
				return train_2.to_csv('./apps/train_ais.csv', header=True,index=False)
	
	#Clear out and sentences with label = 1
	def delete_all_ones():		 		 
		delete_ones=st.checkbox('Delete all training sentences that were previously of interest')
		if delete_ones:
			train_3= train[train.labels == 0]			 
			st.write(":thumbsup:")
			return train_3.to_csv('./apps/train_ais.csv', header=True,index=False)


	#Clear out all sentencs with label = 0
	def delete_all_zeros():	 
		delete_zeros=st.checkbox('Delete all training data that was not previously of interest')
		if delete_zeros:
			train_4= train[train.labels == 1]			 
			st.write(":thumbsup:")
			return train_4.to_csv('./apps/train_ais.csv', header=True,index=False)

	#Delete all data and retur empty datadrame with twoo columns: sentences and label
	def delete_all_data():		 
		delete_everything=st.checkbox('Delete all training data and create new empty dataset')
		if delete_everything:
			train_5= pd.DataFrame(columns=(['sentences','labels']))			 
			st.write(":thumbsup:")
			return train_5.to_csv('./apps/train_ais.csv', header=True,index=False)


	def check_train_data():
			check_train_data=st.checkbox(' Check latest train dataset')
			if check_train_data:
				tr= pd.read_csv('./apps/train_ais.csv')
				st.write(tr)


			

		 
	#Main block of code	 
	train=load_training()	
	new_sentence()
	replace_file()
	delete_all_ones()
	delete_all_zeros()
	delete_all_data()
	check_train_data()
	 
	                
		 
	        
         
    
    

	 

 







		


 

 




 
 