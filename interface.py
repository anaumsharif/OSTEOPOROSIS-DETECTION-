import pickle
import numpy as np
import streamlit as st

# loading model
loaded_model = pickle.load(open("model.pkl", "rb"))

st.header("Meta predictor ")

st.image("banner.jpg")


age=st.number_input("Enter Age ")

drug_type={'None':0, 'Yes':1, 'Ecosprin':2}
drug_history = st.selectbox(
    "Drug History",
    drug_type)

Associtive_type={'No':0 ,'diabetes':1,'BP':2,'diabetes,bp':3,'kidney stone':4,'increase in heart rate':5,'diabetes,heart blockage':6,'diabetes,kidney stone':7}


associtive = st.selectbox("Associative",Associtive_type)


surgery_type={'No':0,'vericose vein surgery':1,'uteres removal':2,'kidney stone opreration':3 ,'uterus surgery':4,'yes(diverticulities)':5,'shouler surgery':6,'knee surgery':7,'yes(open heart surgery)':8}

surgery=st.selectbox("Select Surgery",surgery_type)


gender_type={"Male":0,"Female":1}
gender=st.selectbox("Gender",gender_type)

b1=st.number_input("Bone Density 1")
b2=st.number_input("Bone Density 2")
b3=st.number_input("Bone Density 3")

# 1 Age
# 2 gender
# 3 Associatiove
# 4 surgery
# 5 drug
# 6 frequency

def pred_result():
    age_input=age
    gender_input=gender_type[gender]
    Associative_input=Associtive_type[associtive]
    surgery_input=surgery_type[surgery]
    drug_input=drug_type[drug_history]
    frequency=( b1+b2+b3)/3.0
    to_predict = np.array([age_input,gender_input,Associative_input,surgery_input,drug_input,frequency]).reshape(1,-1)
    pred = loaded_model.predict(to_predict)[0]
    return pred

if st.button('Predict',type="primary"):
    predictions=pred_result()
    print(predictions)
    pred_type={
        1:"Positive",
        0:"Negative"
    }
    t=pred_type[predictions]
    print(predictions)
    if predictions==1:
        
        st.title(f' :red[{t}]:thumbsdown:')
    else:
        st.title(f' :green[{t}]:thumbsup:')
        
    # st.header(f"{t}")
else:
    pass
