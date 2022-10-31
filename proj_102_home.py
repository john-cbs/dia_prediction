import streamlit as st
import numpy as np
import pandas  as pd

def exec(df):
	st.text("""Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.

There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help in reducing the impact of diabetes.

This Web app will help you to predict whether a person has diabetes or is prone to get diabetes in future by analysing the values of several features using the Decision Tree Classifier.
""")
	st.subheader("View Data")
	with st.expander("The Dataset"):
		st.dataframe(df)
	st.subheader("Column Description")
	a,b,c,d = st.columns(4)
	with a:
		if st.checkbox("view column names"):
			st.table(df.columns)
	with b:
		if st.checkbox("view column dtype"):
			types = {i:str(type(df[i][0])) for i in df.columns}
			dtypes_data = pd.DataFrame(data = types.values(),index = types.keys())

			st.dataframe(dtypes_data)
	with c:
		if st.checkbox("view column data"):
			col = st.selectbox("select the column",df.columns)
			st.table(df[col])
	if st.checkbox("view summary"):
	    st.write(df.describe())