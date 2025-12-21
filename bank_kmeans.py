import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# model
model = joblib.load('bank_kmeans.pkl')


# dataframe
df_raw = pd.read_csv("bank_full.csv", sep=";")


# personal information
logo = "parami.jpg"
st.sidebar.image(logo, use_container_width=True)
st.sidebar.markdown(
    """
    **Introduction to Machine Learning
    **Nyein Min Soe
    **Student ID:** PIUS20230027  
    **Email:** nyeinminsoe@parami.edu.mm
    """
)


# input
st.sidebar.header("Input Features")
def user_input_features():
    poutcome = st.sidebar.selectbox("Poutcome", df_raw['poutcome'].unique())
    contact = st.sidebar.selectbox("Contact", df_raw['contact'].unique())
    month = st.sidebar.selectbox("Month", df_raw['month'].unique())
    education = st.sidebar.selectbox("Education", df_raw['education'].unique())
    pdays = st.sidebar.slider("Pdays", int(df_raw['pdays'].min()), int(df_raw['pdays'].max()), 0)
    previous = st.sidebar.number_input("Previous", min_value=0, value=0)
    
    data = {
        'poutcome': poutcome,
        'pdays': pdays,
        'contact': contact,
        'previous': previous,
        'month': month,
        'education': education
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
cluster_pred = model.predict(input_df)


# main title
st.title("Bank Customer Clustering")
st.write(f"### The predicted cluster for the input user is: **Cluster {cluster_pred[0]}**")
st.subheader("Cluster Visualization & Analysis")


# plot
df_raw['cluster'] = model.predict(df_raw)
cluster_df = df_raw[df_raw['cluster'] == cluster_pred[0]]

x_axis = st.selectbox("Select X-axis", cluster_df.columns[:-1], index=0)
y_axis = st.selectbox("Select Y-axis", cluster_df.columns[:-1], index=1)

fig, ax = plt.subplots(figsize=(4, 2.5))
sns.scatterplot(
    data=cluster_df, 
    x=x_axis, 
    y=y_axis, 
    hue='cluster', 
    palette='viridis', 
    style='cluster',
    s=100,
    ax=ax
)
plt.title(f" Visualizing Cluster {cluster_pred[0]} on: {x_axis} vs {y_axis}")
st.pyplot(fig)


# summary
if st.checkbox("Show cluster data summary"):

    st.write(cluster_df.describe(include='all'))
