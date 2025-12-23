import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


model = joblib.load('bank_kmeans.pkl')


col_to_drop = ['y', 'default']
df = pd.read_csv("bank.csv", sep=";").drop(columns=col_to_drop, axis=1)


labels = model.named_steps['kmeans'].labels_
df_clustered = df.copy()
df_clustered['cluster'] = labels
    
num_cols = df.select_dtypes(include=['int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()


# calculate mean and mode
summary = df_clustered.groupby('cluster').agg({
        **{col: 'mean' for col in num_cols},
        **{col: lambda x: x.mode()[0] if not x.mode().empty else "N/A" for col in cat_cols}
    }).reset_index()


# title
st.title("Cluster Details Dashboard")
st.markdown("Detailed breakdown of clusters resulted from the model.")


# Select Cluster
st.sidebar.header("Filter Options")
selected_cluster = st.sidebar.selectbox("Select a Cluster to Inspect", sorted(df_clustered['cluster'].unique()))


# cluster info
m_col1, m_col2 = st.columns(4)

cluster_count = len(df_clustered[df_clustered['cluster'] ==selected_cluster])
cluster_per = cluster_count * 100 / len(df_clustered)

with m_col1:
    st.metric("Customers", f"{len(df_clustered[df_clustered['cluster'] == selected_cluster])}")

with m_col2:
    st.metric("Dataset Coverage", f"{cluster_per:.2f}%")


# Cluster Profile
col1, col2 = st.columns(2)


with col1:
    st.subheader("Categorical Columns")
    # Show categorical modes for the selected cluster
    cluster_modes = df_clustered[df_clustered['cluster'] == selected_cluster].select_dtypes(include='object').mode().T
    cluster_modes.columns = ['Most Frequent Value']
    st.table(cluster_modes)

with col2:
    st.subheader("Numeric Columns")
    # Show numerical averages for the selected cluster
    cluster_avgs = df_clustered[df_clustered['cluster'] == selected_cluster].select_dtypes(include='int').drop(columns='cluster', axis=1).mean().T.to_frame()
    cluster_avgs.columns = ['Mean Value']
    st.table(cluster_avgs)


# plot
st.subheader("Cluster Distribution Plot")
cluster_df = df_clustered[df_clustered['cluster'] == selected_cluster]


plot_cols = [c for c in df.columns if c != 'cluster']
x_axis = st.selectbox("Select X-axis", plot_cols, index=0)
y_axis = st.selectbox("Select Y-axis", plot_cols, index=4)



fig, ax = plt.subplots(figsize=(10, 6))

# background
sns.scatterplot(
    data=df_clustered, 
    x=x_axis, 
    y=y_axis, 
    color='blue', 
    alpha=0.3, 
    ax=ax, 
    label='Other Clusters'
)

# show cluster
cluster_df = df_clustered[df_clustered['cluster'] == selected_cluster]

sns.scatterplot(
    data=cluster_df, 
    x=x_axis, 
    y=y_axis, 
    color='green',
    s=90,
    edgecolor='black', 
    ax=ax, 
    label=f"Cluster {selected_cluster}"
)

plt.title(f"Cluster {selected_cluster}: {x_axis} vs {y_axis}")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
st.pyplot(fig)


























