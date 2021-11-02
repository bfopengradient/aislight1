import numpy as np
import pandas as pd
import streamlit as st

 
"""app allows user to add or delete test data"""


def app():
	st.write('#')
	st.markdown(' ### Edit or replace test data')
	st.write('#')
	

	#Load training data
	def load_test():
		test= pd.read_csv('./apps/test_ais.csv',usecols=['sentences','labels'])
		return test
 
 
	#Update with new sentence and label function 
	def enter_sentence_category(): 
		new_data=[] 
		new={}    
		#need two text fields 
		new['sentences']=st.text_input("Enter sentence for test data set: either something that is bad or innocent")
		new['labels']=st.number_input('Enter label: 1 for sentence of interest and 0 for sentence not of interest', min_value=0, max_value=1)
		new_data.append(new) 
		return new_data

	#Add individual sentence and label 	
	def new_sentence():
		add_sentence= st.checkbox('Add a new sentence and label to the test dataset')
		if add_sentence:
		#create new data frame, align columns with input data and write in append mode to csv
			test_1= pd.DataFrame(columns={'labels','sentences'})
			test_1=test_1.loc[:,['sentences','labels']]
			#call update function and append results to test csv
			test_1= test_1.append(enter_sentence_category(),ignore_index=True) 
			st.write('#')
			add_data= st.button("Update test data ")
			if add_data:
				st.write(":thumbsup:")
				return test_1.to_csv('./apps/test_ais.csv', mode='a', header=False,index=False)

	#Add/upload csv  file 
	def replace_file():	 
		add_file=st.checkbox('Replace current test data csv file with new one ')
		if add_file:
			st.write('Uploaded file needs just two columns: sentences and labels ')			 
			dataset = st.file_uploader("Upload file here", type = ['csv'])
			if dataset is not None:
				test_2 = pd.read_csv(dataset,usecols=['sentences','labels'])
				st.write('Displying top of the uploaded csv file')
				st.dataframe(test_2.head(5))
				st.write(":thumbsup:")
				return test_2.to_csv('./apps/test_ais.csv', header=True,index=False)
	
	#Clear out and sentences with label = 1
	def delete_all_ones():		 	 
		delete_ones=st.checkbox('Delete all test sentences that were previously of interest')
		if delete_ones:
			test_3= test[test.labels == 0]			 
			st.write(":thumbsup:")
			return test_3.to_csv('./apps/test_ais.csv', header=True,index=False)


	#Clear out all sentencs with label = 0
	def delete_all_zeros():	 
		delete_zeros=st.checkbox('Delete all test data that was not previously of interest')
		if delete_zeros:
			test_4= test[test.labels == 1]			 
			st.write(":thumbsup:")
			return test_4.to_csv('./apps/test_ais.csv', header=True,index=False)

	#Delete all data and retur empty datadrame with twoo columns: sentences and label
	def delete_all_data():		 
		delete_everything=st.checkbox('Delete all test data and create new empty dataset')
		if delete_everything:
			test_5= pd.DataFrame(columns=(['sentences','labels']))			 
			st.write(":thumbsup:")
			return test_5.to_csv('./apps/test_ais.csv', header=True,index=False)

	def check_test_data():
		check_test_data=st.checkbox(' Check latest test dataset')
		if check_test_data:
			tst= pd.read_csv('./apps/test_ais.csv')
			st.write(tst)
 
		 
	#Main block of code	 
	test=load_test()	
	new_sentence()
	replace_file()
	delete_all_ones()
	delete_all_zeros()
	delete_all_data()
	check_test_data()

	 
	         