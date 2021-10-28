import streamlit as st
from streamlit_player import st_player
 


""" home page has video explaining the app """
def app(): 
	st.write("#")
	st.markdown(' ### Summary video for AI SUPERVISOR')
	st.write("#")	
	with open("./AISupervisor.mp4", 'rb') as v:
		st.video(v)

 




	 








	 
	     
