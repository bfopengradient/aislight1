from dataset import Dataset
from datasets import load_metric
import numpy as np
import pandas as pd 
import streamlit as st
import torch
import transformers
transformers.logging.set_verbosity_debug()
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification, Trainer, TrainingArguments
 
 

"""app performs hyperparameter optimization with optuna backend and then traininng of model with best hyperparameters"""
	
def app():
	#Rocket git will show when model is training
	gif = "https://aws1.discourse-cdn.com/business7/uploads/streamlit/original/2X/2/247a8220ebe0d7e99dbbd31a2c227dde7767fbe1.gif" 
	 
	#call in training once search is done
	@st.cache(allow_output_mutation=True)
	def model_init():
					return DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', return_dict=True)

	@st.cache(allow_output_mutation=True)
	def get_tokenizer():
		return DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')


	@st.cache(allow_output_mutation=True)
	def load_data():
			TEST_FILE_NAME =    './apps/test_ais.csv'  
			TRAIN_FILE_NAME = './apps/train_ais.csv'
			test = pd.read_csv(TEST_FILE_NAME,encoding='utf8') 
			training = pd.read_csv(TRAIN_FILE_NAME,encoding='utf8')
			return training , test


	#Use for evaluation
	def compute_metrics(eval_pred):
		#metric_acc = load_metric("accuracy")
		metric_f1 = load_metric("f1")
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)
		return metric_f1.compute(predictions=predictions,references=labels)




	#Run a hyperparameter(hps') search with optuna and train model with hp's from highest f1 trial.
	def run_hyperparameter_search():

		#Get tokenizer  
		tokenizer =  get_tokenizer()
				 
		#Get data
		train_texts= list(training.sentences)
		val_texts=list(test.sentences)
		train_labels=list(training.labels)
		val_labels=list(test.labels)

		#Encode train and val sets
		train_encodings = tokenizer(train_texts, truncation=True, padding=True)
		val_encodings = tokenizer(val_texts, truncation=True, padding=True)
		#Create Torch datasets for transformer
		train_dataset = Dataset(train_encodings, train_labels)
		val_dataset = Dataset(val_encodings, val_labels)
		st.write('#')
		st.markdown(" ### Train AI Supervisor")
		st.write('#')
		 
		 

		#Button to check before commencing what is an exhaustive search and train update of the models
		hyper_button=st.button('Start training')
		if hyper_button:
			st.write('#')
			st.markdown('#### Hyperparameter search in progress')
			#gif
			gif_runner = st.image(gif)
			#Define arguments for training
			training_args = TrainingArguments("test", evaluation_strategy="steps", eval_steps=500, disable_tqdm=True)
			trainer = Trainer(
			args=training_args,
			tokenizer=tokenizer,
			train_dataset=train_dataset,
			eval_dataset=val_dataset,
			model_init=model_init,
			compute_metrics=compute_metrics,
			)
			#Run 10 trials of hyperparameter search with optuna
			best_trial=trainer.hyperparameter_search(
			    direction="maximize", 			     
			    n_trials=10 # number of trials
			)

 

			#Pass hyperparameters corresponnding to best f1 to train model and then save model to file locally
			for n, v in best_trial.hyperparameters.items():
				setattr(trainer.args, n, v)
			st.markdown("#### Hyperparameter search complete")
			st.markdown("#### Training in progress with optimal hyperparameters")			 
			trainer.train()
			gif_runner.empty()	
			st.markdown(f"#### F1 Score:{trainer.evaluate()['eval_f1']}")
			st.markdown("#### Training complete")
			st.markdown("#### Model saved to './fine_tuned_model' ")
			trainer.save_model('./fine_tuned_model')

	#Main block of code 
	training,test = load_data()
	run_hyperparameter_search()
