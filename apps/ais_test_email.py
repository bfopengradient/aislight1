from dataset import Dataset 
from datasets import load_metric
import datetime
import email
import imaplib
import mailbox
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
transformers.logging.set_verbosity_error()

import nltk
nltk.download('punkt')
 
 

"""app checks email for content of interest"""
def app():

	st.write('#')
	@st.cache(allow_output_mutation=True)  
	def get_tokenizer():
		return DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
	@st.cache(allow_output_mutation=True)  
	def get_model():
		return DistilBertForSequenceClassification.from_pretrained('./fine_tuned_model')
 
	 
	#Check email and return contents in a dataframe
	st.markdown(' ### Enter details of email you need AI Supervisor to check')
	st.write('#')
	EMAIL_ACCOUNT = st.text_input('Input your email here:',type="password") 
	PASSWORD = st.text_input("Enter a password: ", type="password")
	st.write("#")
	button= st.button('Check Email') 
	if button:
		 
		def process_email():
		    mail = imaplib.IMAP4_SSL('imap.gmail.com')
		    mail.login(EMAIL_ACCOUNT, PASSWORD)
		    mail.list()
		    mail.select('inbox')
		    result, data = mail.uid('search', None, "all")  
		    i = len(data[0].split())
		    X_test_df = pd.DataFrame(columns=['Date', 'From', 'To', 'Subject', 'text'])
		    for x in range(i):
		        latest_email_uid = data[0].split()[x]
		        result, email_data = mail.uid('fetch', latest_email_uid, '(RFC822)')
		        raw_email = email_data[0][1]
		        raw_email_string = raw_email.decode('utf-8')
		        email_message = email.message_from_string(raw_email_string)
		        date_tuple = email.utils.parsedate_tz(email_message['Date'])
		        if date_tuple:
		            local_date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
		            local_message_date = "%s" %(str(local_date.strftime("%a, %d %b %Y %H:%M:%S")))
		        email_from = str(email.header.make_header(email.header.decode_header(email_message['From']))).split()[-1].replace('<','').replace('>','')        
		        email_to = str(email.header.make_header(email.header.decode_header(email_message['To'])))
		        subject = str(email.header.make_header(email.header.decode_header(email_message['Subject'])))

		        for part in email_message.walk():
		            if part.get_content_type() == "text/plain":
		                body = part.get_payload(decode=True)
		                body = body.decode('utf-8')
		                for itm in [[local_message_date, email_from, email_to, subject, final.strip()] for a in body.strip().split('\r\n') for final in a.strip().split('.') if final.strip() !='']:
		                    X_test_df.loc[len(X_test_df)] = itm
		    return X_test_df
			  
 
	    #After retriving contents of email , process and predict if anything is interesting
		def test_model_email():
			st.write('#')
			st.markdown("##### Fetching most recently trained memory")
			 
			tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
			model = DistilBertForSequenceClassification.from_pretrained('./fine_tuned_model')
	  
			st.markdown('##### Checking email inbox for content of interest')
		 
			data=process_email()
			 
			#Get data with labels of interest
			data.loc[:,'label']=1
			 
			#Define x and y
			X= list(data['text'])
			y = list(data['label'].astype(int))
			val_texts_1=list(X)
			val_labels_1=list(y)
			val_encodings_1 = tokenizer(val_texts_1, truncation=True, padding=True)
			val_dataset_1 = Dataset(val_encodings_1, val_labels_1)
			eval_dataloader = DataLoader(val_dataset_1, batch_size=1)
			model.eval()
			from datasets import load_metric
			#Return predictions 
			preds=[]
			for batch in eval_dataloader:
			    batch = {k: v for k, v in batch.items()}
			    with torch.no_grad():
			        outputs = model(**batch)
			    logits = outputs.logits
			    predictions = torch.argmax(logits, dim=-1)		    
			    preds.append(predictions.numpy()[0])		 
			data['model_predictions']= preds

			if data['model_predictions'].any() == 1:
				st.markdown('##### Found some emails of interest')	
				st.write('#')	 		 
				st.markdown('##### Here are the sentences of concern across all emails')

				st.write(data.loc[(data['model_predictions'] ==1),['text']])
				st.write('#')
				st.markdown('##### Details of the emails of concern')
				st.write(data.loc[(data['model_predictions'] ==1),['Date','From','Subject']].drop_duplicates())



		#Call main function
		test_model_email()
   	 

