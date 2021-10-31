from apps import about,training,ais_update_train_data,ais_update_test_data,ais_test_email,pdf_ais
from multiapp import MultiApp 
from PIL import Image
import streamlit as st
 

#Main page that calls the multiapp 

#LOGO_IMAGE  and title positioning 
NAME_IMAGE = "./apps/RGB_Secondary.png"
logo = Image.open(NAME_IMAGE)
st.image(logo, width=700) 
col1, col2, col3 = st.beta_columns([2,2,1])
with col1:
    st.write("")
with col2:
    st.markdown(' ## AI SUPERVISOR')     
with col3:
    st.write( "")
 

#Custom button to match color scheme thats contained in the logo  
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #273346;height:3em;width:15em;border-radius:10px 10px 10px 10px;font-family: 'Segoe UI SemiBold';
  font-size: 20px;
  color: #FFFFFF;
}
</style>""", unsafe_allow_html=True)


 
#Main body of code, uses Multiapp class from multiapp.py file
app = MultiApp()
#Add pages here if needed. Import from apps any additional apps you create and wish to add to AI Supervisor.
app.add_app("About AI Supervisor", about.app)
app.add_app("Train AI Supervisor", training.app)
app.add_app("Update training data", ais_update_train_data.app)
app.add_app("Update test data", ais_update_test_data.app)
app.add_app("Check Email", ais_test_email.app)
app.add_app("Summarize PDF", pdf_ais.app)
 
#Costmetic spacing
st.write('#')
st.write('#')
#Call app.run 
app.run()
 


 
	 

 
 



 

 

