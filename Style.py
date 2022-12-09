import streamlit as st

def set_style():
    st.markdown(
        f'''
            <style>
                .sidebar .sidebar-content {{
                    width: 375px;
                }}
            </style>
        ''',
        unsafe_allow_html=True
    )
