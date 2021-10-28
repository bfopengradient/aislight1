from pdfminer.high_level import extract_text 
import spacy
import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline 
import transformers
transformers.logging.set_verbosity_debug()



"""app checks pdf for named entities of interest and summarizes """
def app():
	st.write('#') 
	st.markdown(' ### Load PDF file for summary and entity check.')
	st.write('#')


	#Get transformer t5 model for summarization
	@st.cache(allow_output_mutation=True) 
	def get_summarizer(text):
		summarizer= pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")
		s= summarizer(text, min_length=5, max_length=500,truncation=True)
		s = str(s).strip('').replace("'summary_text':", "").replace("[{", " ").replace("}]", " ")
		return s 
 
	#Get countbased lexy summarizer(this is significantly quicker than the t5-base model)
	@st.cache(allow_output_mutation=True) 
	def sumy_summarizer(text):
		# Initializing the parser
		my_parser = PlaintextParser.from_string(text,Tokenizer('english'))
		# Creating a summary of 3 sentences.
		lex_rank_summarizer = LexRankSummarizer()
		#limited response to five sentences
		lexrank_summary = lex_rank_summarizer(my_parser.document,sentences_count=5)
		# return the summary 
		for sentence in lexrank_summary:
		  return str(sentence)

  	#Using spacy for ner. selecting only proper nouns although any part of speach or dependency could be searched for if needed.
	@st.cache(allow_output_mutation=True) 
	def spacy_ner(text):
		nlp = spacy.load("en_core_web_sm")
		doc = nlp(text)
		proper_nouns=[]	 
		for token in doc:
		    if token.pos_ == "PROPN" and token.dep_ == "pobj":
		        #print(token.text, token.pos_, token.dep_)
		        proper_nouns.append(token.text)		    
		unique= set(proper_nouns) 
		return str(unique).replace("{", " ").replace("}", " ") 


	#transformer ner model
	@st.cache(allow_output_mutation=True) 
	#@st.cache(suppress_st_warning=True)
	def transformer_ner(text):		 
		ner = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER",grouped_entities=True)	 
		doc = ner(text)
		proper_nouns=[]	 
		for entity in doc:  		 
			proper_nouns.append(entity['word'])
		unique_transformer= set(proper_nouns) 
		#dealing with sub word tokenization and decided to remove and partial tokens that get through group_entity net above 
		full_words=[]
		for token in unique_transformer:
			if not token.startswith('##'):
				if int(len(token) >3):
					full_words.append(token)
		return str(full_words).replace("[", " ").replace("]", " ") 


	#Button to execute ner and summarization		 	 
	uploaded_pdf = st.file_uploader("Load pdf: ", type=['pdf'])
	if uploaded_pdf :
		text = extract_text(uploaded_pdf)
		t2=text.replace(",", "").replace("\n","")
		st.write('#')
		st.markdown(" #### Quick summary/entity check.") 
		quick_summary= sumy_summarizer(t2)
		st.markdown("##### Summary: "), st.write(str(quick_summary))
		unique=spacy_ner(t2)		
		st.markdown(" ##### Entities: "), st.write(str(unique))
		st.write('#')
		st.markdown(' #### Summary/entity check using deep learning memory.')
		st.markdown(" ##### Summary: ") , st.write(get_summarizer(t2))
		transformer_ner=transformer_ner(t2)		
		st.markdown(" ##### Entities: "),st.write(str(transformer_ner))

					 
	 


				         



		 


		 
