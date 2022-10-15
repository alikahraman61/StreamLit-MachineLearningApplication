import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pokemon=pd.read_csv('pokemon.csv')
combat=pd.read_csv('combats.csv')

def idgetir(name):
    df=pokemon.set_index("Name")
    id=df.loc[name]['#']
    return id

pokemonlist=list(pokemon['Name'].unique())
col1,col2=st.columns(2)

with col1:
    poke1=st.selectbox("First Pokemon",pokemonlist)
    poke1=poke1.replace("Mega ","")
    poke1 = poke1.replace(" X", "")
    poke1 = poke1.replace(" Y", "")
    poke1=poke1.replace("♂","-m")
    poke1=poke1.replace("♀","-f")
    link1="images/"+poke1.lower()+".png"
    st.image(link1)

with col2:
    poke2 = st.selectbox("Seccond Pokemon", pokemonlist)
    poke2 = poke2.replace("Mega ", "")
    poke2 = poke2.replace(" X", "")
    poke2 = poke2.replace(" Y", "")
    poke2 = poke2.replace("♂", "-m")
    poke2 = poke2.replace("♀", "-f")
    link2 = "images/" + poke2.lower() + ".png"
    st.image(link2)

#### SİDEBAR


cdf=combat
cdf['Winner']=cdf['First_pokemon']==cdf['Winner']
cdf['Winner']=np.where(cdf['Winner'],0,1)
y=cdf[['Winner']]
x=cdf.drop("Winner",axis=1)

trainsec=st.sidebar.slider("Train Size",0,100,80)
trainsec=trainsec/100

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=trainsec)

modelsec=st.sidebar.selectbox("Model Seç",["Decision Tree","Random Forest"])

if modelsec=="Decision Tree":
    tree=DecisionTreeClassifier()
    model=tree.fit(x_train,y_train)

if modelsec=="Random Forest":
    agacsec=st.sidebar.number_input("Agacsayisi",value=100)
    forest=RandomForestClassifier(n_estimators=agacsec)
    model=forest.fit(x_train,y_train)

col1,col2,col3=st.columns(3)
with col1:
    pass
with col2:
    saldir = st.button("Müsabaka Başlasın")
with col3:
    pass

if saldir:
   sonuc= model.predict([[idgetir(poke1),idgetir(poke2)]])
   st.write(int(sonuc))
   if sonuc==0:
       st.header("Kazanan")
       st.image(link1)
       st.write("Model Skoru",model.score(x_test, y_test))
   else:
       st.header("Kazanan")
       st.image(link2)
       st.write("Model Skoru",model.score(x_test, y_test))

