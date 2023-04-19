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

st.write("---")
st.write(f"This is a web calculator to estimate someone's probability of needing Post Mastectomy Radiation Treatment (PMRT), before pathology results are available.")
st.write(f"To use this calculator, fill in the pre-operative information about the patient to calculate the probability of PMRT.")
st.write("Note: for each risk factor below, the average value among our patient cohort will be used by default, unless you enter a value. I.e., if a risk factor is missing, the average value in the dataset used to construct this calculator will be used.")

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
# st.markdown(".stTextInput > label {font-size:105%; font-weight:bold; color:blue;} ",unsafe_allow_html=True) #for all text-input label sections
# st.markdown(".stMultiSelect > label {font-size:105%; font-weight:bold; color:blue;} ",unsafe_allow_html=True) #for all multi-select label sections
# st.write("---")
# st.write(f"The model shown is based on the Logistic Lasso algorithm. It found {len(std_coef_df) - 1} PMRT risk factors to be the most relevant predictors for PMRT.")
st.write(f"We display the factors in the order of decreasing importance found by the model.")
st.write('---')
st.write("## Calculator Form")
with st.form(key='my_form'):
    
    # For each column in the unstandardized df, create a form element that corresponds to the dtype
    for i, row in unstd_coef_df.iterrows():

        col_name = row["Feature"]
        if col_name == "intercept":
            # st.write(f"Intercept: {row['Coefficient']:.5f}")
            continue
        if col_name == "PRE_surg_indicat_prim___recurrent_cancer":
            val = 0
            form_to_val[col_name] = val
            continue
        if col_name == "PRE_surg_indicat_prim___primary_tx":
            val = 1
            form_to_val[col_name] = val
            continue
        st.write("---")
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
            avg_val = col_to_avg[col_name]
            val = st.slider(label=label, key=col_name, max_value=100.0, step=0.1, value=avg_val)
            html_str = f"<span style='color:grey'>If the information is missing or unknown, you could choose to use the average value in dataset ({col_to_avg[col_name]:.3f}) or provide a best guess. </span>"
            st.markdown(html_str, unsafe_allow_html=True)
        elif dtype == "integer":
            val = st.slider(label=label, key=col_name, step=1, max_value=100, value=avg_val)
        else:
            st.write(f"Error: dtype {dtype} not recognised")
            continue
        if val == -1:
            # write text in grey
            options_str_to_show = options_str.replace("-1, missing |", "")
            html_str = f"<span style='color:grey'>If the information is missing or unknown, select 'missing', and the average value in dataset ({col_to_avg[col_name]:.3f}) will be used. <br>{options_str_to_show}</span>"
            st.markdown(html_str, unsafe_allow_html=True)
            val = round(col_to_avg[col_name], 5)
        # html_str = f"<span style='color:grey'>Value of {col_name}: {val}</span>"
        # st.markdown(html_str, unsafe_allow_html=True)
        form_to_val[col_name] = val
        
    st.write("---")
    submit_button = form_submit_button('Calculate Probability')
    st.write("To reset the form, please refresh the page.")
    st.write("For support, please contact ANONYMOUS.SUBMISSION@EXAMPLE.COM")
form_values = form_to_val

def my_round(x, base=10):
    return base * round(x/base)

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
        rounded_prob = my_round(prob, base=0.1)
        st.title(f"The probability of needing PMRT is approximately {int(rounded_prob*100)}%.")
        st.write("---")
        st.markdown("### Values entered:")
        st.write(form_values)
        # write out the calculation
        # st.write("---")
        # st.markdown("### Calculation (unstandardized coefficient * value):")
        # for col_name, unstd_coef in col_to_unstd_coef.items():
        #     if col_name == "intercept":
        #         st.caption(f"{unstd_coef:.3f} (intercept)")
        #         continue
        #     form_val = form_values[col_name]
        #     # st.write(f"\+ {unstd_coef:.5f} * {form_val} ({col_name})")
        #     st.caption(f"\+ {unstd_coef:.5f} * {form_val} ({VarReader.read_var_attrib(col_name, has_missing=False)['label']})")
        
        # st.write(f"= {score:.5f}")
        # st.write("---")
        # st.markdown(f"### Estimated Probability: 1 / (1 + exp(- ({score:.5f}))) ≈ {prob*100:.1f}% ≈ {int(rounded_prob*100)}%")
        # st.write("The final step used the logit function, which transforms the output of a linear regression model into a probability (between 0 and 1). We round to probability the nearest 10%.")
        # st.write("---")
    
        
    set_style()