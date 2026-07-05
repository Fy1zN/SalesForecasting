import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# ---------------------------
# Data Loading (cached)
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv', encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    return df

df = load_data()

def get_season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title(" Navigation")
page = st.sidebar.radio("Go to", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])

# ---------------------------
# PAGE 1: Sales Overview Dashboard
# ---------------------------
if page == "Sales Overview":
    st.title(" Sales Overview Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Sales by Year")
        yearly = df.groupby('Order Year')['Sales'].sum().reset_index()
        fig = px.bar(yearly, x='Order Year', y='Sales', color='Order Year')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Sales Trend")
        monthly = df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
        fig2 = px.line(monthly, x='Order Date', y='Sales', markers=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sales by Region and Category (Filterable)")
    col3, col4 = st.columns(2)
    with col3:
        region_filter = st.multiselect("Filter by Region", df['Region'].unique(), default=list(df['Region'].unique()))
    with col4:
        category_filter = st.multiselect("Filter by Category", df['Category'].unique(), default=list(df['Category'].unique()))

    filtered = df[(df['Region'].isin(region_filter)) & (df['Category'].isin(category_filter))]
    region_cat = filtered.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
    fig3 = px.bar(region_cat, x='Region', y='Sales', color='Category', barmode='group')
    st.plotly_chart(fig3, use_container_width=True)
    
    
    # ---------------------------
# PAGE 2: Forecast Explorer
# ---------------------------
elif page == "Forecast Explorer":
    st.title(" Forecast Explorer")

    dimension = st.selectbox("Select Dimension", ["Category", "Region"])
    if dimension == "Category":
        options = df['Category'].unique()
    else:
        options = df['Region'].unique()

    selected_value = st.selectbox(f"Select {dimension}", options)
    horizon = st.slider("Forecast Horizon (months ahead)", 1, 3, 3)

    @st.cache_data
    def forecast_segment(dimension, selected_value, periods=3):
        seg_df = df[df[dimension] == selected_value].copy()
        seg_monthly = seg_df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
        seg_monthly.columns = ['Month', 'Sales']

        seg_monthly['Lag1'] = seg_monthly['Sales'].shift(1)
        seg_monthly['Lag2'] = seg_monthly['Sales'].shift(2)
        seg_monthly['Lag3'] = seg_monthly['Sales'].shift(3)
        seg_monthly['RollingMean3'] = seg_monthly['Sales'].shift(1).rolling(window=3).mean()
        seg_monthly['MonthNum'] = seg_monthly['Month'].dt.month
        seg_monthly['Quarter'] = seg_monthly['Month'].dt.quarter
        seg_monthly['Season'] = seg_monthly['MonthNum'].apply(get_season)

        seg_monthly = seg_monthly.dropna().reset_index(drop=True)

        feature_cols = ['Lag1', 'Lag2', 'Lag3', 'RollingMean3', 'MonthNum', 'Quarter', 'Season']
        X = seg_monthly[feature_cols]
        y = seg_monthly['Sales']

        X_train, X_test = X.iloc[:-3], X.iloc[-3:]
        y_train, y_test = y.iloc[:-3], y.iloc[-3:]

        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return seg_monthly, y_test, y_pred, mae, rmse

    seg_monthly, y_test, y_pred, mae, rmse = forecast_segment(dimension, selected_value)

    forecast_months = seg_monthly['Month'].iloc[-3:].iloc[:horizon]
    forecast_values = y_pred[:horizon]
    actual_values = y_test.values[:horizon]

    fig = px.line(seg_monthly, x='Month', y='Sales', title=f"{selected_value} — Sales & Forecast")
    fig.add_scatter(x=forecast_months, y=forecast_values, mode='lines+markers', name='Forecast', line=dict(color='red'))
    fig.add_scatter(x=forecast_months, y=actual_values, mode='lines+markers', name='Actual', line=dict(color='green'))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    
    # ---------------------------
# PAGE 3: Anomaly Report
# ---------------------------
elif page == "Anomaly Report":
    st.title(" Anomaly Report")

    @st.cache_data
    def detect_anomalies():
        weekly = df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()
        weekly.columns = ['Week', 'Sales']

        iso_model = IsolationForest(contamination=0.1, random_state=42)
        weekly['Anomaly_IF'] = iso_model.fit_predict(weekly[['Sales']])
        weekly['Anomaly_IF'] = weekly['Anomaly_IF'].map({1: 'Normal', -1: 'Anomaly'})

        return weekly

    weekly = detect_anomalies()
    anomalies = weekly[weekly['Anomaly_IF'] == 'Anomaly']

    fig = px.line(weekly, x='Week', y='Sales', title="Weekly Sales with Anomalies")
    fig.add_scatter(x=anomalies['Week'], y=anomalies['Sales'], mode='markers', 
                     marker=dict(color='red', size=10), name='Anomaly')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomalies")
    st.dataframe(anomalies[['Week', 'Sales']].reset_index(drop=True), use_container_width=True)
    
    # ---------------------------
# PAGE 4: Product Demand Segments
# ---------------------------
elif page == "Product Demand Segments":
    st.title(" Product Demand Segments")

    @st.cache_data
    def build_clusters():
        subcat_sales = df.groupby('Sub-Category')['Sales'].sum().reset_index()
        subcat_sales.columns = ['Sub-Category', 'Total_Sales']

        subcat_yearly = df.groupby(['Sub-Category', 'Order Year'])['Sales'].sum().reset_index()
        subcat_pivot = subcat_yearly.pivot(index='Sub-Category', columns='Order Year', values='Sales')
        first_year, last_year = subcat_pivot.columns.min(), subcat_pivot.columns.max()
        subcat_pivot['Growth_Rate'] = ((subcat_pivot[last_year] - subcat_pivot[first_year]) / subcat_pivot[first_year]) * 100

        subcat_monthly = df.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='ME')])['Sales'].sum().reset_index()
        subcat_volatility = subcat_monthly.groupby('Sub-Category')['Sales'].std().reset_index()
        subcat_volatility.columns = ['Sub-Category', 'Volatility']

        subcat_aov = df.groupby('Sub-Category')['Sales'].mean().reset_index()
        subcat_aov.columns = ['Sub-Category', 'Avg_Order_Value']

        features_df = subcat_sales.merge(subcat_pivot[['Growth_Rate']], on='Sub-Category')
        features_df = features_df.merge(subcat_volatility, on='Sub-Category')
        features_df = features_df.merge(subcat_aov, on='Sub-Category')

        X_cluster = features_df[['Total_Sales', 'Growth_Rate', 'Volatility', 'Avg_Order_Value']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        features_df['Cluster'] = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        features_df['PCA1'] = X_pca[:,0]
        features_df['PCA2'] = X_pca[:,1]

        return features_df

    features_df = build_clusters()

    cluster_labels = {0: "High Growth, High Volatility Outlier", 1: "Low Volume, Stable Demand", 2: "High Volume, High Value Core"}
    features_df['Cluster Label'] = features_df['Cluster'].map(cluster_labels)

    fig = px.scatter(features_df, x='PCA1', y='PCA2', color='Cluster Label', 
                      text='Sub-Category', title="Product Sub-Category Clusters")
    fig.update_traces(textposition='top center', marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Categories by Cluster")
    st.dataframe(features_df[['Sub-Category', 'Cluster Label', 'Total_Sales', 'Growth_Rate', 'Volatility', 'Avg_Order_Value']], use_container_width=True)