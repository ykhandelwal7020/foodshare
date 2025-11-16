import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             r2_score, mean_squared_error, mean_absolute_error,
                             silhouette_score, davies_bouldin_score)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx

# Page configuration
st.set_page_config(
    page_title="UAE Food Tiffin Analytics",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data
def generate_synthetic_data(n_samples=600):
    """Generate synthetic survey data for demonstration"""
    np.random.seed(42)
    
    data = {
        'Q1_Age': np.random.choice(['18-24 years', '25-34 years', '35-44 years', '45-54 years', '55+ years'],
                                   n_samples, p=[0.10, 0.45, 0.30, 0.10, 0.05]),
        'Q2_Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.60, 0.40]),
        'Q3_Nationality': np.random.choice(['Indian', 'Pakistani', 'Filipino', 'Arab (UAE/GCC)', 'Western'],
                                          n_samples, p=[0.35, 0.15, 0.12, 0.23, 0.15]),
        'Q4_Employment_Status': np.random.choice(['Full-time employed', 'Part-time employed', 'Self-employed', 'Student'],
                                                 n_samples, p=[0.65, 0.15, 0.15, 0.05]),
        'Q5_Monthly_Income': np.random.choice(['Below 5,000', '5,000 - 10,000', '10,001 - 15,000',
                                               '15,001 - 25,000', '25,001 - 40,000', 'Above 40,000'],
                                              n_samples, p=[0.08, 0.15, 0.22, 0.30, 0.18, 0.07]),
        'Q6_Household_Size': np.random.choice(['Living alone', '2 people', '3-4 people', '5+ people'],
                                              n_samples, p=[0.25, 0.30, 0.35, 0.10]),
        'Q9_Ordering_Frequency': np.random.choice(['Daily', '4-6 times per week', '2-3 times per week',
                                                   'Once a week', 'Rarely'],
                                                  n_samples, p=[0.12, 0.25, 0.33, 0.20, 0.10]),
        'Q10_Weekly_Spending': np.random.choice(['Less than 100', '100 - 200', '201 - 300', '301 - 500', 'Above 500'],
                                                n_samples, p=[0.15, 0.30, 0.30, 0.18, 0.07]),
    }
    
    # Importance ratings (1-5)
    for col in ['Taste_Quality', 'Price_Affordability', 'Health_Hygiene', 'Delivery_Speed',
                'Variety_Options', 'Homecooked_Style', 'Authenticity_Cuisine']:
        data[f'Q12_{col}'] = np.random.choice([3, 4, 5], n_samples, p=[0.20, 0.45, 0.35])
    
    # Interest level (correlated with income)
    interest_levels = []
    for income in data['Q5_Monthly_Income']:
        if income in ['Above 40,000', '25,001 - 40,000']:
            interest = np.random.choice(['Extremely Interested', 'Very Interested', 'Moderately Interested'],
                                       p=[0.40, 0.40, 0.20])
        elif income in ['15,001 - 25,000', '10,001 - 15,000']:
            interest = np.random.choice(['Very Interested', 'Moderately Interested', 'Slightly Interested'],
                                       p=[0.30, 0.50, 0.20])
        else:
            interest = np.random.choice(['Moderately Interested', 'Slightly Interested', 'Not Interested'],
                                       p=[0.30, 0.40, 0.30])
        interest_levels.append(interest)
    data['Q21_Interest_Level'] = interest_levels
    
    # Willingness to spend (correlated with income and interest)
    spending_levels = []
    for income, interest in zip(data['Q5_Monthly_Income'], data['Q21_Interest_Level']):
        if income == 'Above 40,000' and interest in ['Extremely Interested', 'Very Interested']:
            spend = np.random.choice(['31 - 40 AED', '41 - 50 AED', 'Above 50 AED'], p=[0.30, 0.40, 0.30])
        elif income in ['25,001 - 40,000', '15,001 - 25,000']:
            spend = np.random.choice(['21 - 25 AED', '26 - 30 AED', '31 - 40 AED'], p=[0.30, 0.50, 0.20])
        else:
            spend = np.random.choice(['15 - 20 AED', '21 - 25 AED', '26 - 30 AED'], p=[0.40, 0.40, 0.20])
        spending_levels.append(spend)
    data['Q23_Willingness_to_Spend_Per_Meal'] = spending_levels
    
    # Food preferences (Italian)
    italian_items = ['Margherita Pizza', 'Pasta Carbonara', 'Pasta Alfredo', 'Lasagna', 'Garlic Bread']
    data['Q13_Italian_Food_Preferences'] = ['; '.join(np.random.choice(italian_items, 
                                            np.random.randint(2, 4), replace=False)) 
                                            for _ in range(n_samples)]
    
    # Food preferences (Chinese)
    chinese_items = ['Fried Rice', 'Hakka Noodles', 'Manchurian', 'Spring Rolls', 'Chilli Chicken']
    data['Q14_Chinese_Food_Preferences'] = ['; '.join(np.random.choice(chinese_items, 
                                            np.random.randint(2, 4), replace=False)) 
                                            for _ in range(n_samples)]
    
    df = pd.DataFrame(data)
    return df

def convert_df_to_excel(df):
    """Convert dataframe to Excel bytes for download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

def preprocess_data(df, target_col=None, task='classification'):
    """
    Preprocess data for ML models.
    NOTE: This function now ONLY handles encoding, not scaling.
    Scaling must be done *after* train-test split to prevent data leakage.
    """
    df = df.copy()
    
    # Select relevant features
    feature_cols = [col for col in df.columns if col not in [target_col, 'Response_ID'] 
                   and not col.startswith('Q13_') and not col.startswith('Q14_')]
    
    X = df[feature_cols].copy()
    
    # Encode categorical variables
    le_dict = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Handle target variable
    le_target = None  # <-- FIX: Initialize le_target to None
    
    if target_col:
        y = df[target_col].copy()
        if task == 'classification':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        elif task == 'regression':
            # Convert spending to numeric
            spending_map = {
                'Less than 15 AED': 12.5, '15 - 20 AED': 17.5, '21 - 25 AED': 23.0,
                '26 - 30 AED': 28.0, '31 - 40 AED': 35.5, '41 - 50 AED': 45.5,
                'Above 50 AED': 60.0,
                # Adding potential mappings from the synthetic data generation
                '31 - 40 AED': 35.5, '41 - 50 AED': 45.5, 'Above 50 AED': 60.0,
                '21 - 25 AED': 23.0, '26 - 30 AED': 28.0, '31 - 40 AED': 35.5,
                '15 - 20 AED': 17.5, '21 - 25 AED': 23.0, '26 - 30 AED': 28.0
            }
            # Fill missing values before mapping
            y = y.map(spending_map).fillna(y.mode()[0] if not y.empty else 28.0)
            
    else:
        y = None
        # le_target is already None
    
    # --- SCALING REMOVED FROM THIS FUNCTION ---
    # Scaling will be done after train-test split
    
    return X, y, le_dict, le_target

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🍱 UAE FOOD TIFFIN SERVICE - Analytics Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/250x100/667eea/ffffff?text=Food+Tiffin", 
                use_column_width=True)
        st.markdown("### 📊 Data Upload")
        
        data_source = st.radio("Choose data source:", 
                              ["Use Sample Data", "Upload CSV", "Load from GitHub"])
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows")
            else:
                df = generate_synthetic_data()
                st.info("Using sample data")
        elif data_source == "Load from GitHub":
            github_url = st.text_input("Enter GitHub raw CSV URL:", 
                                      "https://raw.githubusercontent.com/...")
            if st.button("Load Data"):
                try:
                    df = pd.read_csv(github_url)
                    st.success(f"✅ Loaded {len(df)} rows from GitHub")
                except:
                    st.error("Failed to load. Using sample data.")
                    df = generate_synthetic_data()
            else:
                df = generate_synthetic_data()
        else:
            df = generate_synthetic_data()
            st.success(f"✅ Sample data: {len(df)} rows")
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        st.metric("Total Records", len(df))
        st.metric("Features", len(df.columns))
        
        st.markdown("---")
        st.markdown("""
        ### 📖 About
        This dashboard provides:
        - Association Rule Mining
        - Classification Models
        - Customer Clustering
        - Regression Analysis
        - Dynamic Pricing Tool
        """)
    
    # Main tabs
    tabs = st.tabs(["🔗 Association Rules", "🎯 Classification", "👥 Clustering", 
                    "📈 Regression", "💰 Dynamic Pricing"])
    
    # ========================================================================
    # TAB 1: ASSOCIATION RULE MINING
    # ========================================================================
    with tabs[0]:
        st.header("🔗 Association Rule Mining - Food Preferences")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_support = st.slider("Minimum Support", 0.01, 0.5, 0.10, 0.01)
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.70, 0.05)
        with col3:
            min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.5, 0.1)
        
        if st.button("🚀 Run Association Analysis", key='assoc_button'):
            with st.spinner("Mining association rules..."):
                try:
                    # Prepare transaction data
                    transactions = []
                    for idx, row in df.iterrows():
                        items = []
                        if 'Q13_Italian_Food_Preferences' in df.columns:
                            italian = str(row['Q13_Italian_Food_Preferences']).split('; ')
                            items.extend([f"Italian: {item}" for item in italian if item != 'nan'])
                        if 'Q14_Chinese_Food_Preferences' in df.columns:
                            chinese = str(row['Q14_Chinese_Food_Preferences']).split('; ')
                            items.extend([f"Chinese: {item}" for item in chinese if item != 'nan'])
                        if items:
                            transactions.append(items)
                    
                    # Convert to binary matrix
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    df_binary = pd.DataFrame(te_ary, columns=te.columns_)
                    
                    # Apply Apriori
                    frequent_itemsets = apriori(df_binary, min_support=min_support, use_colnames=True)
                    
                    if len(frequent_itemsets) > 0:
                        # Generate rules
                        rules = association_rules(frequent_itemsets, metric="confidence", 
                                                 min_threshold=min_confidence)
                        rules = rules[rules['lift'] >= min_lift]
                        
                        if len(rules) > 0:
                            rules = rules.sort_values('lift', ascending=False)
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Frequent Itemsets", len(frequent_itemsets))
                            col2.metric("Association Rules", len(rules))
                            col3.metric("Avg Lift", f"{rules['lift'].mean():.2f}")
                            
                            # Top rules table
                            st.subheader("📊 Top 10 Association Rules")
                            display_rules = rules.head(10).copy()
                            display_rules['antecedents'] = display_rules['antecedents'].apply(
                                lambda x: ', '.join(list(x)[:2]))
                            display_rules['consequents'] = display_rules['consequents'].apply(
                                lambda x: ', '.join(list(x)))
                            st.dataframe(
                                display_rules[['antecedents', 'consequents', 'support', 
                                             'confidence', 'lift']].style.format({
                                    'support': '{:.3f}',
                                    'confidence': '{:.3f}',
                                    'lift': '{:.2f}'
                                }),
                                use_container_width=True
                            )
                            
                            # Visualizations
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("📈 Top Rules by Lift")
                                fig = px.bar(rules.head(15), x='lift', y=rules.head(15).index,
                                           orientation='h', color='confidence',
                                           color_continuous_scale='Viridis')
                                fig.update_layout(height=400, showlegend=False,
                                                yaxis_title="Rule Index", xaxis_title="Lift")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.subheader("🎯 Support vs Confidence")
                                fig = px.scatter(rules, x='support', y='confidence', size='lift',
                                               color='lift', hover_data=['lift'],
                                               color_continuous_scale='Reds')
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Network graph
                            if len(rules) >= 5:
                                st.subheader("🕸️ Rules Network Graph")
                                
                                G = nx.DiGraph()
                                top_rules = rules.head(20)
                                
                                for idx, row in top_rules.iterrows():
                                    antecedents = list(row['antecedents'])[:1]
                                    consequents = list(row['consequents'])[:1]
                                    for ant in antecedents:
                                        for cons in consequents:
                                            G.add_edge(ant[:30], cons[:30], weight=row['lift'])
                                
                                pos = nx.spring_layout(G, k=2, iterations=50)
                                
                                edge_trace = []
                                for edge in G.edges():
                                    x0, y0 = pos[edge[0]]
                                    x1, y1 = pos[edge[1]]
                                    edge_trace.append(
                                        go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                                  mode='lines', line=dict(width=0.5, color='#888'),
                                                  hoverinfo='none', showlegend=False)
                                    )
                                
                                node_trace = go.Scatter(
                                    x=[pos[node][0] for node in G.nodes()],
                                    y=[pos[node][1] for node in G.nodes()],
                                    mode='markers+text',
                                    text=[node for node in G.nodes()],
                                    textposition="top center",
                                    hoverinfo='text',
                                    marker=dict(size=20, color='lightblue', 
                                              line=dict(width=2, color='darkblue'))
                                )
                                
                                fig = go.Figure(data=edge_trace + [node_trace])
                                fig.update_layout(showlegend=False, hovermode='closest',
                                                height=500, xaxis=dict(showgrid=False, zeroline=False),
                                                yaxis=dict(showgrid=False, zeroline=False))
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Download button
                            st.markdown("---")
                            rules_export = rules.copy()
                            rules_export['antecedents'] = rules_export['antecedents'].apply(
                                lambda x: ', '.join(list(x)))
                            rules_export['consequents'] = rules_export['consequents'].apply(
                                lambda x: ', '.join(list(x)))
                            
                            st.download_button(
                                label="📥 Download Association Rules (CSV)",
                                data=rules_export.to_csv(index=False).encode('utf-8'),
                                file_name='association_rules.csv',
                                mime='text/csv'
                            )
                        else:
                            st.warning("No rules found with current thresholds. Try lowering the parameters.")
                    else:
                        st.warning("No frequent itemsets found. Try lowering minimum support.")
                
                except Exception as e:
                    st.error(f"Error in association analysis: {str(e)}")
    
    # ========================================================================
    # TAB 2: CLASSIFICATION
    # ========================================================================
    with tabs[1]:
        st.header("🎯 Classification - Customer Interest Prediction")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            target_col = 'Q21_Interest_Level'
            st.info(f"**Target Variable:** {target_col}")
        
        with col2:
            binary_mode = st.checkbox("Convert to Binary Classification", value=False)
        
        # Model selection
        st.subheader("⚙️ Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            models_selected = st.multiselect(
                "Select Models to Compare:",
                ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"],
                default=["Logistic Regression", "Random Forest"]
            )
        with col2:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        
        if st.button("🚀 Train Classification Models", key='class_button'):
            with st.spinner("Training models..."):
                try:
                    # 1. Preprocess data (encoding only)
                    X, y, le_dict, le_target = preprocess_data(df, target_col, 'classification')
                    
                    # 2. Binary conversion if needed
                    if binary_mode:
                        # Assuming 'Not Interested' and 'Slightly Interested' are 0, others are 1
                        # This depends on the LabelEncoder's fit. A safer way:
                        not_interested_labels = ['Not Interested', 'Slightly Interested']
                        interested_classes_idx = [i for i, cls in enumerate(le_target.classes_) 
                                                  if cls not in not_interested_labels]
                        y = np.where(np.isin(y, interested_classes_idx), 1, 0)
                        class_names = ['Not/Slightly Interested', 'Interested']
                    else:
                        class_names = le_target.classes_
                    
                    # 3. Train-test split (on unscaled data)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # 4. Scale data *after* splitting
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # 5. Train models on scaled data
                    results = {}
                    
                    if "Logistic Regression" in models_selected:
                        lr = LogisticRegression(max_iter=1000, random_state=42)
                        lr.fit(X_train_scaled, y_train)
                        results['Logistic Regression'] = {
                            'model': lr,
                            'pred': lr.predict(X_test_scaled),
                            'proba': lr.predict_proba(X_test_scaled)
                        }
                    
                    if "Random Forest" in models_selected:
                        rf = RandomForestClassifier(n_estimators=100, random_state=42)
                        rf.fit(X_train_scaled, y_train)
                        results['Random Forest'] = {
                            'model': rf,
                            'pred': rf.predict(X_test_scaled),
                            'proba': rf.predict_proba(X_test_scaled)
                        }
                    
                    if "Gradient Boosting" in models_selected:
                        from sklearn.ensemble import GradientBoostingClassifier
                        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
                        gb.fit(X_train_scaled, y_train)
                        results['Gradient Boosting'] = {
                            'model': gb,
                            'pred': gb.predict(X_test_scaled),
                            'proba': gb.predict_proba(X_test_scaled)
                        }
                    
                    if "SVM" in models_selected:
                        svm = SVC(probability=True, random_state=42)
                        svm.fit(X_train_scaled, y_train)
                        results['SVM'] = {
                            'model': svm,
                            'pred': svm.predict(X_test_scaled),
                            'proba': svm.predict_proba(X_test_scaled)
                        }
                    
                    # Model comparison
                    st.subheader("📊 Model Performance Comparison")
                    
                    comparison_data = []
                    for name, res in results.items():
                        acc = accuracy_score(y_test, res['pred'])
                        prec = precision_score(y_test, res['pred'], average='weighted')
                        rec = recall_score(y_test, res['pred'], average='weighted')
                        f1 = f1_score(y_test, res['pred'], average='weighted')
                        comparison_data.append({
                            'Model': name, 'Accuracy': acc, 'Precision': prec,
                            'Recall': rec, 'F1-Score': f1
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.dataframe(comparison_df.style.format({
                            'Accuracy': '{:.4f}', 'Precision': '{:.4f}',
                            'Recall': '{:.4f}', 'F1-Score': '{:.4f}'
                        }), use_container_width=True)
                    
                    with col2:
                        fig = px.bar(comparison_df, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                   barmode='group', title='Model Metrics Comparison')
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Best model details
                    best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
                    best_model_res = results[best_model_name]
                    
                    st.success(f"🏆 Best Model: **{best_model_name}** with {comparison_df['Accuracy'].max():.2%} accuracy")
                    
                    # Confusion Matrix
                    st.subheader("🎯 Confusion Matrix (Best Model)")
                    cm = confusion_matrix(y_test, best_model_res['pred'])
                    
                    fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                                   x=class_names, y=class_names, color_continuous_scale='Blues')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Classification Report
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_test, best_model_res['pred'], 
                                                  target_names=class_names, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                    
                    # ROC Curves (for binary classification)
                    if binary_mode:
                        st.subheader("📈 ROC Curves")
                        fig = go.Figure()
                        
                        for name, res in results.items():
                            fpr, tpr, _ = roc_curve(y_test, res['proba'][:, 1])
                            roc_auc = auc(fpr, tpr)
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={roc_auc:.3f})',
                                                    mode='lines'))
                        
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random',
                                                mode='lines', line=dict(dash='dash')))
                        fig.update_layout(xaxis_title='False Positive Rate',
                                        yaxis_title='True Positive Rate',
                                        height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature Importance
                    if hasattr(best_model_res['model'], 'feature_importances_'):
                        st.subheader("📊 Feature Importance")
                        importance = best_model_res['model'].feature_importances_
                        importance_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                   color='Importance', color_continuous_scale='Viridis')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download predictions
                    st.markdown("---")
                    predictions_df = pd.DataFrame({
                        'Actual': y_test,
                        'Predicted': best_model_res['pred'],
                        'Correct': y_test == best_model_res['pred']
                    })
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=predictions_df.to_csv(index=False).encode('utf-8'),
                            file_name='classification_predictions.csv',
                            mime='text/csv'
                        )
                    with col2:
                        st.download_button(
                            label="📥 Download Metrics (Excel)",
                            data=convert_df_to_excel(comparison_df),
                            file_name='classification_metrics.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                
                except Exception as e:
                    st.error(f"Error in classification: {str(e)}")
    
    # ========================================================================
    # TAB 3: CLUSTERING
    # ========================================================================
    with tabs[2]:
        st.header("👥 Customer Clustering & Segmentation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        with col2:
            algorithm = st.selectbox("Clustering Algorithm", 
                                    ["K-Means", "Hierarchical", "DBSCAN"])
        with col3:
            if algorithm == "DBSCAN":
                eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("DBSCAN min_samples", 2, 10, 5)
        
        if st.button("🚀 Run Clustering Analysis", key='cluster_button'):
            with st.spinner("Performing clustering..."):
                try:
                    # Prepare features for clustering
                    feature_cols = [col for col in df.columns 
                                  if col not in ['Q13_Italian_Food_Preferences', 
                                               'Q14_Chinese_Food_Preferences']]
                    X_cluster = df[feature_cols].copy()
                    
                    # Encode and scale
                    for col in X_cluster.select_dtypes(include=['object']).columns:
                        le = LabelEncoder()
                        X_cluster[col] = le.fit_transform(X_cluster[col].astype(str))
                    
                    scaler = StandardScaler()
                    # For clustering (unsupervised), it's standard to fit on all data
                    X_scaled = scaler.fit_transform(X_cluster)
                    
                    # Apply clustering
                    if algorithm == "K-Means":
                        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels = model.fit_predict(X_scaled)
                    elif algorithm == "Hierarchical":
                        model = AgglomerativeClustering(n_clusters=n_clusters)
                        labels = model.fit_predict(X_scaled)
                    else:  # DBSCAN
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = model.fit_predict(X_scaled)
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    # Calculate metrics
                    if n_clusters > 1:
                        silhouette = silhouette_score(X_scaled, labels)
                        davies_bouldin = davies_bouldin_score(X_scaled, labels)
                    else:
                        silhouette = -1
                        davies_bouldin = -1
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Clusters Found", n_clusters)
                    col2.metric("Silhouette Score", f"{silhouette:.3f}")
                    col3.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}")
                    
                    # Cluster distribution
                    st.subheader("📊 Cluster Distribution")
                    cluster_counts = pd.Series(labels).value_counts().sort_index()
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                                   labels={'x': 'Cluster', 'y': 'Count'},
                                   color=cluster_counts.values, color_continuous_scale='Viridis')
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                                   title='Cluster Size Distribution')
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Elbow curve (for K-Means)
                    if algorithm == "K-Means":
                        st.subheader("📈 Elbow Curve")
                        inertias = []
                        silhouettes = []
                        K_range = range(2, 11)
                        
                        for k in K_range:
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            kmeans.fit(X_scaled)
                            inertias.append(kmeans.inertia_)
                            silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers'))
                            fig.update_layout(title='Elbow Curve', xaxis_title='Number of Clusters',
                                            yaxis_title='Inertia', height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=list(K_range), y=silhouettes, mode='lines+markers'))
                            fig.update_layout(title='Silhouette Score', xaxis_title='Number of Clusters',
                                            yaxis_title='Silhouette Score', height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # PCA visualization
                    st.subheader("🎨 2D PCA Visualization")
                    pca = PCA(n_components=2, random_state=42)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    pca_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Cluster': labels
                    })
                    
                    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                                   title=f'PCA Scatter Plot ({pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%} = {sum(pca.explained_variance_ratio_):.1%} variance)',
                                   color_continuous_scale='Viridis' if algorithm == "DBSCAN" else None)
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster profiles
                    st.subheader("👤 Cluster Profiles (Customer Personas)")
                    
                    df_with_clusters = df.copy()
                    df_with_clusters['Cluster'] = labels
                    
                    # Filter out noise points for DBSCAN
                    if algorithm == "DBSCAN":
                        df_with_clusters = df_with_clusters[df_with_clusters['Cluster'] != -1]
                    
                    profile_data = []
                    for cluster_id in sorted(df_with_clusters['Cluster'].unique()):
                        cluster_df = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
                        
                        profile = {
                            'Cluster': f"Cluster {cluster_id}",
                            'Size': len(cluster_df),
                            '% of Total': f"{len(cluster_df)/len(df_with_clusters)*100:.1f}%"
                        }
                        
                        # Add key characteristics
                        for col in ['Q1_Age', 'Q5_Monthly_Income', 'Q21_Interest_Level', 
                                  'Q23_Willingness_to_Spend_Per_Meal']:
                            if col in cluster_df.columns:
                                mode_val = cluster_df[col].mode()
                                profile[col.replace('Q', '').replace('_', ' ')] = mode_val[0] if len(mode_val) > 0 else 'N/A'
                        
                        profile_data.append(profile)
                    
                    profile_df = pd.DataFrame(profile_data)
                    st.dataframe(profile_df, use_container_width=True)
                    
                    # Download results
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Cluster Assignments (CSV)",
                            data=df_with_clusters.to_csv(index=False).encode('utf-8'),
                            file_name='cluster_assignments.csv',
                            mime='text/csv'
                        )
                    with col2:
                        st.download_button(
                            label="📥 Download Cluster Profiles (Excel)",
                            data=convert_df_to_excel(profile_df),
                            file_name='cluster_profiles.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                
                except Exception as e:
                    st.error(f"Error in clustering: {str(e)}")
    
    # ========================================================================
    # TAB 4: REGRESSION
    # ========================================================================
    with tabs[3]:
        st.header("📈 Regression - Willingness to Pay Prediction")
        
        target_col = 'Q23_Willingness_to_Spend_Per_Meal'
        st.info(f"**Target Variable:** {target_col} (converted to numeric)")
        
        # Model selection
        col1, col2 = st.columns(2)
        with col1:
            reg_models_selected = st.multiselect(
                "Select Regression Models:",
                ["Linear Regression", "Ridge", "Lasso", "Random Forest", "Gradient Boosting"],
                default=["Linear Regression", "Random Forest"]
            )
        with col2:
            test_size_reg = st.slider("Test Set Size (%) ", 10, 40, 20, key='reg_test_size') / 100
        
        if st.button("🚀 Train Regression Models", key='reg_button'):
            with st.spinner("Training regression models..."):
                try:
                    # 1. Preprocess data (encoding only)
                    # The '_' discards the le_target, which is None in regression task
                    X, y, le_dict, _ = preprocess_data(df, target_col, 'regression')
                    
                    # Handle potential NaN in y after mapping
                    y = y.fillna(y.median())

                    # 2. Train-test split (on unscaled data)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_reg, random_state=42
                    )
                    
                    # 3. Scale data *after* splitting
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # 4. Train models on scaled data
                    reg_results = {}
                    
                    if "Linear Regression" in reg_models_selected:
                        lr = LinearRegression()
                        lr.fit(X_train_scaled, y_train)
                        reg_results['Linear Regression'] = {
                            'model': lr,
                            'pred': lr.predict(X_test_scaled)
                        }
                    
                    if "Ridge" in reg_models_selected:
                        ridge = Ridge(alpha=1.0, random_state=42)
                        ridge.fit(X_train_scaled, y_train)
                        reg_results['Ridge'] = {
                            'model': ridge,
                            'pred': ridge.predict(X_test_scaled)
                        }
                    
                    if "Lasso" in reg_models_selected:
                        lasso = Lasso(alpha=1.0, random_state=42)
                        lasso.fit(X_train_scaled, y_train)
                        reg_results['Lasso'] = {
                            'model': lasso,
                            'pred': lasso.predict(X_test_scaled)
                        }
                    
                    if "Random Forest" in reg_models_selected:
                        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_reg.fit(X_train_scaled, y_train)
                        reg_results['Random Forest'] = {
                            'model': rf_reg,
                            'pred': rf_reg.predict(X_test_scaled)
                        }
                    
                    if "Gradient Boosting" in reg_models_selected:
                        gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        gb_reg.fit(X_train_scaled, y_train)
                        reg_results['Gradient Boosting'] = {
                            'model': gb_reg,
                            'pred': gb_reg.predict(X_test_scaled)
                        }
                    
                    # Model comparison
                    st.subheader("📊 Model Performance Comparison")
                    
                    comparison_data = []
                    for name, res in reg_results.items():
                        r2 = r2_score(y_test, res['pred'])
                        rmse = np.sqrt(mean_squared_error(y_test, res['pred']))
                        mae = mean_absolute_error(y_test, res['pred'])
                        mape = np.mean(np.abs((y_test - res['pred']) / y_test)) * 100
                        
                        comparison_data.append({
                            'Model': name, 'R²': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.dataframe(comparison_df.style.format({
                            'R²': '{:.4f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'MAPE (%)': '{:.2f}'
                        }), use_container_width=True)
                    
                    with col2:
                        fig = px.bar(comparison_df, x='Model', y=['R²', 'RMSE', 'MAE'],
                                   barmode='group', title='Model Metrics Comparison')
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Best model details
                    best_model_name = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
                    best_model_res = reg_results[best_model_name]
                    
                    st.success(f"🏆 Best Model: **{best_model_name}** with R² = {comparison_df['R²'].max():.4f}")
                    
                    # Predicted vs Actual
                    st.subheader("🎯 Predicted vs Actual Spending")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.scatter(x=y_test, y=best_model_res['pred'],
                                       labels={'x': 'Actual Spending (AED)', 'y': 'Predicted Spending (AED)'},
                                       title='Predicted vs Actual')
                        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                               y=[y_test.min(), y_test.max()],
                                               mode='lines', name='Perfect Prediction',
                                               line=dict(dash='dash', color='red')))
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        residuals = y_test - best_model_res['pred']
                        fig = px.scatter(x=best_model_res['pred'], y=residuals,
                                       labels={'x': 'Predicted Spending (AED)', 'y': 'Residuals'},
                                       title='Residual Plot')
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance
                    if hasattr(best_model_res['model'], 'feature_importances_'):
                        st.subheader("📊 Feature Importance")
                        importance = best_model_res['model'].feature_importances_
                        importance_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                   color='Importance', color_continuous_scale='Viridis')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif hasattr(best_model_res['model'], 'coef_'):
                        st.subheader("📊 Feature Coefficients")
                        coef_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Coefficient': best_model_res['model'].coef_
                        }).sort_values('Coefficient', key=abs, ascending=False).head(15)
                        
                        fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                                   color='Coefficient', color_continuous_scale='RdBu')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download predictions
                    st.markdown("---")
                    predictions_df = pd.DataFrame({
                        'Actual_Spending': y_test,
                        'Predicted_Spending': best_model_res['pred'],
                        'Error': y_test - best_model_res['pred'],
                        'Absolute_Error': np.abs(y_test - best_model_res['pred'])
                    })
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=predictions_df.to_csv(index=False).encode('utf-8'),
                            file_name='regression_predictions.csv',
                            mime='text/csv'
                        )
                    with col2:
                        st.download_button(
                            label="📥 Download Metrics (Excel)",
                            data=convert_df_to_excel(comparison_df),
                            file_name='regression_metrics.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    
                    # 5. Store CORRECT model and scaler in session state
                    st.session_state['best_reg_model'] = best_model_res['model']
                    st.session_state['scaler'] = scaler # This is the scaler fit ONLY on X_train
                    st.session_state['feature_columns'] = X.columns.tolist()
                    st.session_state['le_dict'] = le_dict
                
                except Exception as e:
                    st.error(f"Error in regression: {str(e)}")
    
    # ========================================================================
    # TAB 5: DYNAMIC PRICING
    # ========================================================================
    with tabs[4]:
        st.header("💰 Dynamic Pricing Tool")
        
        st.markdown("""
        This tool predicts the optimal price point for a customer based on their profile.
        Adjust the customer characteristics below to see real-time price predictions.
        """)
        
        if 'best_reg_model' not in st.session_state:
            st.warning("⚠️ Please train a regression model first in the Regression tab!")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👤 Customer Profile")
                
                age = st.selectbox("Age Group", 
                                  ['18-24 years', '25-34 years', '35-44 years', '45-54 years', '55+ years'])
                gender = st.selectbox("Gender", ['Male', 'Female'])
                nationality = st.selectbox("Nationality",
                                         ['Indian', 'Pakistani', 'Filipino', 'Arab (UAE/GCC)', 'Western'])
                employment = st.selectbox("Employment Status",
                                        ['Full-time employed', 'Part-time employed', 'Self-employed', 'Student'])
                income = st.selectbox("Monthly Income (AED)",
                                    ['Below 5,000', '5,000 - 10,000', '10,001 - 15,000',
                                     '15,001 - 25,000', '25,001 - 40,000', 'Above 40,000'])
                household = st.selectbox("Household Size",
                                       ['Living alone', '2 people', '3-4 people', '5+ people'])
            
            with col2:
                st.subheader("🍽️ Behavior & Preferences")
                
                ordering_freq = st.selectbox("Current Ordering Frequency",
                                            ['Daily', '4-6 times per week', '2-3 times per week',
                                             'Once a week', 'Rarely'])
                weekly_spending = st.selectbox("Current Weekly Spending (AED)",
                                              ['Less than 100', '100 - 200', '201 - 300',
                                               '301 - 500', 'Above 500'])
                
                taste_importance = st.slider("Importance: Taste & Quality", 1, 5, 4)
                price_importance = st.slider("Importance: Price/Affordability", 1, 5, 3)
                health_importance = st.slider("Importance: Health & Hygiene", 1, 5, 4)
                speed_importance = st.slider("Importance: Delivery Speed", 1, 5, 3)
                variety_importance = st.slider("Importance: Variety", 1, 5, 3)
                homecooked_importance = st.slider("Importance: Home-cooked Style", 1, 5, 4)
                authenticity_importance = st.slider("Importance: Authenticity", 1, 5, 4)
            
            if st.button("💡 Calculate Optimal Price", key='pricing_button'):
                with st.spinner("Calculating optimal price..."):
                    try:
                        # Create feature dictionary
                        customer_features = {
                            'Q1_Age': age,
                            'Q2_Gender': gender,
                            'Q3_Nationality': nationality,
                            'Q4_Employment_Status': employment,
                            'Q5_Monthly_Income': income,
                            'Q6_Household_Size': household,
                            'Q9_Ordering_Frequency': ordering_freq,
                            'Q10_Weekly_Spending': weekly_spending,
                            'Q12_Taste_Quality': taste_importance,
                            'Q12_Price_Affordability': price_importance,
                            'Q12_Health_Hygiene': health_importance,
                            'Q12_Delivery_Speed': speed_importance,
                            'Q12_Variety_Options': variety_importance,
                            'Q12_Homecooked_Style': homecooked_importance,
                            'Q12_Authenticity_Cuisine': authenticity_importance
                        }
                        
                        # Create dataframe
                        customer_df = pd.DataFrame([customer_features])
                        
                        # Encode categorical variables
                        le_dict = st.session_state['le_dict']
                        for col, le in le_dict.items():
                            if col in customer_df.columns:
                                try:
                                    # Ensure the value is in the learned classes
                                    if customer_df[col].values[0] in le.classes_:
                                        customer_df[col] = le.transform(customer_df[col].astype(str))
                                    else:
                                        # Handle unseen category: assign a default (e.g., 0)
                                        customer_df[col] = 0 
                                except:
                                    customer_df[col] = 0  # Default if unseen category
                        
                        # Ensure all feature columns are present and in correct order
                        for col in st.session_state['feature_columns']:
                            if col not in customer_df.columns:
                                customer_df[col] = 0 # Add missing columns (likely numeric)
                        
                        customer_df = customer_df[st.session_state['feature_columns']]
                        
                        # Scale features using the *correct* scaler from session state
                        customer_scaled = st.session_state['scaler'].transform(customer_df)
                        
                        # Predict
                        predicted_price = st.session_state['best_reg_model'].predict(customer_scaled)[0]
                        
                        # Calculate confidence intervals (simplified)
                        confidence = 0.95
                        std_error = 3.0  # Approximate standard error
                        margin = 1.96 * std_error
                        lower_bound = max(12.5, predicted_price - margin)
                        upper_bound = min(60.0, predicted_price + margin)
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 🎯 Pricing Recommendation")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("#### Predicted Price")
                            st.markdown(f"<h1 style='text-align: center; color: #2ecc71;'>{predicted_price:.2f} AED</h1>",
                                      unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("#### Lower Bound (95% CI)")
                            st.markdown(f"<h2 style='text-align: center;'>{lower_bound:.2f} AED</h2>",
                                      unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("#### Upper Bound (95% CI)")
                            st.markdown(f"<h2 style='text-align: center;'>{upper_bound:.2f} AED</h2>",
                                      unsafe_allow_html=True)
                        
                        # Pricing strategy
                        st.markdown("---")
                        st.markdown("### 📋 Pricing Strategy")
                        
                        if predicted_price >= 35:
                            segment = "Premium"
                            color = "#2ecc71"
                            recommendation = f"""
                            **Customer Segment: {segment}**
                            
                            This customer has high willingness to pay. Recommendations:
                            - Offer premium menu items (35-50 AED range)
                            - Highlight quality, authenticity, and gourmet options
                            - Provide exclusive chef selections
                            - Emphasize convenience and time-saving
                            - Consider monthly premium subscription (1500-2000 AED)
                            """
                        elif predicted_price >= 25:
                            segment = "Standard"
                            color = "#3498db"
                            recommendation = f"""
                            **Customer Segment: {segment}**
                            
                            This customer is in the standard range. Recommendations:
                            - Offer standard menu items (25-35 AED range)
                            - Balance between quality and value
                            - Promote weekly meal plans with discounts
                            - Highlight home-cooked authentic flavors
                            - Consider weekly subscription (800-1200 AED/month)
                            """
                        else:
                            segment = "Budget"
                            color = "#e67e22"
                            recommendation = f"""
                            **Customer Segment: {segment}**
                            
                            This customer is price-sensitive. Recommendations:
                            - Offer budget-friendly options (15-24 AED range)
                            - Emphasize value for money and affordability
                            - Promote combo meals and group orders
                            - Offer referral discounts
                            - Consider daily/weekly subscriptions with high discounts
                            """
                        
                        st.markdown(f"<div style='padding: 20px; background-color: {color}20; border-left: 5px solid {color}; border-radius: 5px;'>{recommendation}</div>",
                                  unsafe_allow_html=True)
                        
                        # Revenue projection
                        st.markdown("---")
                        st.markdown("### 💰 Revenue Projection")
                        
                        orders_per_week = st.slider("Expected Orders per Week", 1, 7, 3)
                        
                        weekly_revenue = predicted_price * orders_per_week
                        monthly_revenue = weekly_revenue * 4
                        annual_revenue = monthly_revenue * 12
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Weekly Revenue", f"{weekly_revenue:.2f} AED")
                        col2.metric("Monthly Revenue", f"{monthly_revenue:.2f} AED")
                        col3.metric("Annual Revenue", f"{annual_revenue:.2f} AED")
                        
                        # Demand factors adjustment
                        st.markdown("---")
                        st.markdown("### ⚙️ Dynamic Pricing Adjustments")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            peak_hours = st.checkbox("Peak Hours (+10%)")
                            weekend = st.checkbox("Weekend (+5%)")
                            first_time = st.checkbox("First-Time Customer (-15%)")
                        
                        with col2:
                            loyalty = st.checkbox("Loyalty Member (-10%)")
                            bulk_order = st.checkbox("Bulk Order (3+ meals) (-12%)")
                            promotional = st.checkbox("Promotional Period (-20%)")
                        
                        # Calculate adjusted price
                        adjusted_price = predicted_price
                        adjustments = []
                        
                        if peak_hours:
                            adjusted_price *= 1.10
                            adjustments.append("Peak Hours: +10%")
                        if weekend:
                            adjusted_price *= 1.05
                            adjustments.append("Weekend: +5%")
                        if first_time:
                            adjusted_price *= 0.85
                            adjustments.append("First-Time: -15%")
                        if loyalty:
                            adjusted_price *= 0.90
                            adjustments.append("Loyalty: -10%")
                        if bulk_order:
                            adjusted_price *= 0.88
                            adjustments.append("Bulk Order: -12%")
                        if promotional:
                            adjusted_price *= 0.80
                            adjustments.append("Promotional: -20%")
                        
                        if adjustments:
                            st.markdown(f"**Final Adjusted Price: {adjusted_price:.2f} AED**")
                            st.markdown("**Applied Adjustments:**")
                            for adj in adjustments:
                                st.markdown(f"- {adj}")
                        
                    except Exception as e:
                        st.error(f"Error in price calculation: {str(e)}")

if __name__ == "__main__":
    main()