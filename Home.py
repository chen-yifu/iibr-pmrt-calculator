import streamlit as st
import pandas as pd
import numpy as np
from os.path import join
from VarReader import VarReader
from Style import set_style
from streamlit import form_submit_button

DATA_DIR = "data/"

st.title('Post-Mastectomy Radiation Treatment Probability Calculator')
# ask user to select between Logistic Lasso and Elastic Net
model_to_use = "Logistic Lasso" # = st.selectbox("Select model", ["Logistic Lasso", "Elastic Net"])
std_coef_df_path = join(DATA_DIR, f"standardized_coef_{model_to_use}.csv")
unstd_coef_df_path = join(DATA_DIR, f"unstandardized_coef_{model_to_use}.csv")
metadata_path = join(DATA_DIR, "Metadata.xlsx")
col_avg_path = join(DATA_DIR, "col_to_avg.csv")


std_coef_df = pd.read_csv(std_coef_df_path, sep=",")
unstd_coef_df = pd.read_csv(unstd_coef_df_path, sep=",")
col_avg = pd.read_csv(col_avg_path)

st.write(f"The model used is based on Logistic Lasso. There are {len(std_coef_df)} PMRT risk factors to be considered.")
st.write(f"Fill in the following information about the patient to calculate the probability of PMRT.")
st.write("Note 1: if a value is missing, the average value will be used by default since the model is linear.")
st.write(
    "Note 2: the model is trained on dataset that is a retrospective cohort of consecutive patients who have undergone"
    " mastectomy with immediate alloplastic breast reconstruction from 2010 to 2020"
    # "therefore the expected use case is on patients who would be planned for mastectomy with immediate alloplastic breast reconstruction"
    )

# rename columns to "Feature" and "Coefficient"
std_coef_df.columns = ["Feature", "Coefficient"]
unstd_coef_df.columns = ["Feature", "Coefficient"]
col_avg.columns = ["Feature", "Average"]

# assert features are same in both df
assert set(std_coef_df["Feature"].values) == set(unstd_coef_df["Feature"].values), f"Features are not the same in both df: {set(unstd_coef_df['Feature'].values) - set(std_coef_df['Feature'].values)}"

# Sort the features in unstd_coef_df by the coefficient

std_coef_df = std_coef_df.sort_values(by="Coefficient", ascending=False)
feature_to_idx = {}
for i, row in std_coef_df.iterrows():
    feature_to_idx[row["Feature"]] = i

unstd_coef_df["idx"] = unstd_coef_df["Feature"].apply(lambda x: feature_to_idx[x])
unstd_coef_df = unstd_coef_df.sort_values(by="idx")

col_to_avg = {}
for i, row in col_avg.iterrows():
    col_to_avg[row["Feature"]] = row["Average"]

VarReader = VarReader(metadata_path)

    
form_to_val = {}

with st.form(key='my_form'):
    
    # For each column in the unstandardized df, create a form element that corresponds to the dtype
    for i, row in unstd_coef_df.iterrows():
        st.write("---")
        col_name = row["Feature"]
        if col_name == "intercept":
            # st.write(f"Intercept: {row['Coefficient']:.5f}")
            continue
        # Read the variable attributes from the metadata
        var_attrib = VarReader.read_var_attrib(col_name, has_missing=True)
        section = var_attrib["section"]
        dtype = var_attrib["dtype"]
        label = var_attrib["label"]
        options = var_attrib["options"]
        options_str = var_attrib["options_str"]
        label = f"{label}"
        # Create the form element
        if dtype == "categorical" or dtype == "ordinal":
            description_to_val = {f"{options_str.split('|')[i]}": option for i, option in enumerate(options)}
            val_to_description = {option: f"{str(options_str.split('|')[i]).split(',')[1]}" for i, option in enumerate(options)}
            # for k, v in description_to_val.items():
            #     if 
            val = st.radio(label=label, options=options, key=col_name, horizontal=True, format_func=lambda x: val_to_description[x])
        elif dtype == "checkbox":
            val = st.checkbox(label=label, key=col_name)
        elif dtype == "real":
            val = st.slider(label=label, key=col_name, min_value=-1.0, max_value=100.0, step=0.01, value=-1.0)
        elif dtype == "integer":
            val = st.slider(label=label, key=col_name, step=1, min_value=-1, max_value=100, value=-1)
        else:
            st.write(f"Error: dtype {dtype} not recognised")
            continue
        if val == -1:
            # write text in grey
            html_str = f"<span style='color:grey'>If factor is unsure/missing/unknown, the average value in dataset ({col_to_avg[col_name]:.5f}) will be used</span>"
            st.markdown(html_str, unsafe_allow_html=True)
            val = round(col_to_avg[col_name], 5)
        # html_str = f"<span style='color:grey'>Value of {col_name}: {val}</span>"
        # st.markdown(html_str, unsafe_allow_html=True)
        form_to_val[col_name] = val
        
    st.write("---")
    submit_button = form_submit_button('Submit')

form_values = form_to_val

col_to_unstd_coef = {}
for i, row in unstd_coef_df.iterrows():
    col_to_unstd_coef[row["Feature"]] = row["Coefficient"]

if submit_button:

    with st.sidebar:
        score = 0
        for col_name, unstd_coef in col_to_unstd_coef.items():
            if col_name == "intercept":
                score += unstd_coef
                continue
            form_val = form_values[col_name]
            score += unstd_coef * form_val

        prob = 1 / (1 + np.exp(-score))
        st.title(f"Probability of PMRT: {prob*100:.5}%")
        print("---")
        st.write("Values entered:")
        st.write(form_values)
        # write out the calculation
        st.write("Calculation (unstandardized coefficient * value):")
        for col_name, unstd_coef in col_to_unstd_coef.items():
            if col_name == "intercept":
                st.write(f"{unstd_coef:.3f} (intercept)")
                continue
            form_val = form_values[col_name]
            # st.write(f"\+ {unstd_coef:.5f} * {form_val} ({col_name})")
            st.write(f"\+ {unstd_coef:.5f} * {form_val} ({VarReader.read_var_attrib(col_name, has_missing=False)['label']})")
        
        st.write(f"= {score:.5f}")
        st.write(f"Probaility = 1 / (1 + exp(- ({score:.5f}))) = {prob:.5f}")
        
        st.write("---")
    set_style()