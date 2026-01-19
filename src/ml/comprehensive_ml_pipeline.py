"""
Advanced Healthcare Machine Learning Pipeline
Comprehensive ML models for healthcare analytics with detailed explanations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, silhouette_score
)
from sklearn.impute import SimpleImputer
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class HealthcareMLPipeline:
    """
    Comprehensive machine learning pipeline for healthcare analytics
    Includes 15+ algorithms with detailed explanations and interpretations
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.ml_explanations = self._load_ml_explanations()
    
    def _load_ml_explanations(self):
        """Load comprehensive ML explanations for healthcare context"""
        return {
            "readmission_prediction": {
                "business_problem": "Predicting which patients are likely to be readmitted within 30 days helps hospitals provide better discharge planning and reduce costs.",
                "why_important": "30-day readmissions are a key quality metric that affects hospital reimbursement. Preventing unnecessary readmissions improves patient outcomes and reduces healthcare costs.",
                "target_audience": "Clinical teams, discharge planners, case managers, and quality improvement staff",
                "interpretation": "Higher probability scores indicate patients who need enhanced discharge planning, follow-up care, or transitional care management."
            },
            "length_of_stay_prediction": {
                "business_problem": "Predicting how long patients will stay helps with bed management, staffing, and discharge planning.",
                "why_important": "Accurate LOS predictions improve hospital operations, reduce costs, and ensure appropriate resource allocation.",
                "target_audience": "Bed management, nursing supervisors, case managers, and hospital administrators",
                "interpretation": "Predicted days help plan staffing levels, discharge timing, and identify patients who may need extended care coordination."
            },
            "high_cost_prediction": {
                "business_problem": "Identifying patients likely to incur high healthcare costs enables proactive care management and resource allocation.",
                "why_important": "High-cost patients often have complex conditions requiring coordinated care. Early identification allows for preventive interventions.",
                "target_audience": "Care managers, population health teams, and health plan administrators",
                "interpretation": "High-risk scores indicate patients who would benefit from care management programs, disease management, or enhanced monitoring."
            },
            "mortality_risk_prediction": {
                "business_problem": "Predicting mortality risk helps prioritize critical care resources and inform family discussions.",
                "why_important": "Early identification of high-mortality-risk patients enables appropriate care intensity and family communication.",
                "target_audience": "ICU teams, hospitalists, palliative care, and clinical leadership",
                "interpretation": "Higher risk scores indicate patients needing intensive monitoring, family discussions, or palliative care consultation."
            },
            "patient_segmentation": {
                "business_problem": "Grouping patients with similar characteristics enables targeted interventions and personalized care approaches.",
                "why_important": "Patient segmentation supports population health management, care pathway development, and resource optimization.",
                "target_audience": "Population health managers, care coordinators, and clinical program developers",
                "interpretation": "Different segments represent distinct patient populations requiring tailored care approaches and interventions."
            }
        }
    
    def prepare_readmission_data(self, patients_df, encounters_df, claims_df):
        """Prepare data for readmission prediction model"""
        
        st.info("""
        **🎯 Readmission Prediction Model**
        
        **What we're predicting**: Whether a patient will return to the hospital within 30 days of discharge
        
        **Why this matters**: 
        - Readmissions cost hospitals money due to Medicare penalties
        - High readmission rates indicate potential quality issues
        - Early identification allows for better discharge planning
        
        **How we use the results**:
        - Patients with high readmission risk get enhanced discharge planning
        - Care coordinators provide additional follow-up
        - Social workers address barriers to successful discharge
        """)
        
        # Merge datasets for comprehensive features
        encounter_features = encounters_df.merge(patients_df, on='patient_id', how='left')
        
        # Create target variable (readmission within 30 days)
        encounter_features['target_readmission'] = encounter_features['is_readmission'].astype(int)
        
        # Feature engineering with healthcare domain knowledge
        features = []
        
        # Demographic features
        features.extend(['age', 'gender', 'race'])
        
        # Clinical complexity features
        features.extend(['comorbidity_count', 'risk_score'])
        
        # Encounter-specific features
        features.extend(['length_of_stay', 'severity_level', 'encounter_type'])
        
        # Chronic condition flags
        features.extend(['is_diabetic', 'is_hypertensive', 'has_chronic_disease'])
        
        # Create derived features
        encounter_features['age_risk_interaction'] = encounter_features['age'] * encounter_features['risk_score']
        encounter_features['comorbidity_age_ratio'] = encounter_features['comorbidity_count'] / (encounter_features['age'] + 1)
        encounter_features['los_severity_interaction'] = encounter_features['length_of_stay'] * (
            encounter_features['severity_level'].map({'Minor': 1, 'Moderate': 2, 'Major': 3, 'Critical': 4}).fillna(2)
        )
        
        features.extend(['age_risk_interaction', 'comorbidity_age_ratio', 'los_severity_interaction'])
        
        # Prepare final dataset
        model_data = encounter_features[features + ['target_readmission']].copy()
        
        # Handle missing values
        numeric_features = model_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = model_data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from feature lists
        if 'target_readmission' in numeric_features:
            numeric_features.remove('target_readmission')
        
        # Impute missing values
        if numeric_features:
            numeric_imputer = SimpleImputer(strategy='median')
            model_data[numeric_features] = numeric_imputer.fit_transform(model_data[numeric_features])
        
        if categorical_features:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            model_data[categorical_features] = categorical_imputer.fit_transform(model_data[categorical_features])
        
        return model_data, features
    
    def train_readmission_models(self, model_data, features):
        """Train multiple models for readmission prediction with detailed explanations"""
        
        X = model_data[features]
        y = model_data['target_readmission']
        
        # Encode categorical variables
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_features:
            # Use one-hot encoding for categorical variables
            X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        else:
            X_encoded = X.copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['readmission'] = scaler
        
        # Train multiple models
        models_to_train = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'explanation': """
                **Logistic Regression** is like a smart calculator that weighs different factors to predict probability.
                
                **How it works**: It looks at each patient characteristic (age, diagnosis, etc.) and assigns a weight 
                based on how much that factor increases or decreases readmission risk.
                
                **Strengths**: Easy to interpret, shows which factors are most important, provides probability scores.
                **Best for**: Understanding which patient characteristics drive readmissions.
                """
            },
            'Random Forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'explanation': """
                **Random Forest** is like asking 100 doctors for their opinion and taking the majority vote.
                
                **How it works**: It creates many decision trees, each looking at different combinations of patient 
                factors, then combines their predictions for a final answer.
                
                **Strengths**: Very accurate, handles complex patterns, less likely to overfit.
                **Best for**: Getting the most accurate predictions when you have lots of data.
                """
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'explanation': """
                **Gradient Boosting** learns from its mistakes to get better predictions.
                
                **How it works**: It starts with a simple prediction, then builds additional models to correct 
                the errors from previous models, gradually improving accuracy.
                
                **Strengths**: Often the most accurate, good at finding subtle patterns.
                **Best for**: Maximum prediction accuracy, especially with complex patient populations.
                """
            },
            'Neural Network': {
                'model': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
                'explanation': """
                **Neural Network** mimics how the brain processes information through interconnected neurons.
                
                **How it works**: Information flows through layers of artificial neurons, each learning different 
                patterns in the data, similar to how doctors develop clinical intuition.
                
                **Strengths**: Can learn very complex patterns, adapts to new data well.
                **Best for**: Large datasets with complex relationships between patient factors.
                """
            },
            'Support Vector Machine': {
                'model': SVC(probability=True, random_state=42),
                'explanation': """
                **Support Vector Machine** finds the best boundary between patients who will and won't be readmitted.
                
                **How it works**: It draws an invisible line (or surface) that best separates high-risk from 
                low-risk patients based on their characteristics.
                
                **Strengths**: Works well with smaller datasets, good at handling complex boundaries.
                **Best for**: Situations where you need robust predictions with limited data.
                """
            }
        }
        
        results = {}
        
        for model_name, model_info in models_to_train.items():
            st.write(f"### 🤖 Training {model_name}")
            st.write(model_info['explanation'])
            
            # Train model
            model = model_info['model']
            
            if model_name in ['Logistic Regression', 'Support Vector Machine', 'Neural Network']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Display results with healthcare interpretation
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}", help="Overall correctness of predictions")
                st.metric("Precision", f"{precision:.3f}", help="Of patients predicted to be readmitted, how many actually were")
                st.metric("Recall", f"{recall:.3f}", help="Of patients who were readmitted, how many did we catch")
            
            with col2:
                st.metric("F1 Score", f"{f1:.3f}", help="Balance between precision and recall")
                st.metric("AUC Score", f"{auc:.3f}", help="Ability to distinguish between readmitted and not readmitted patients")
            
            # Healthcare interpretation
            st.write("**🏥 Clinical Interpretation:**")
            if precision > 0.8:
                st.success(f"**High Precision ({precision:.1%})**: When this model flags a patient for readmission risk, it's usually correct. Good for targeting interventions.")
            elif precision > 0.6:
                st.warning(f"**Moderate Precision ({precision:.1%})**: Some false alarms, but still useful for identifying high-risk patients.")
            else:
                st.error(f"**Low Precision ({precision:.1%})**: Many false alarms. May overwhelm care teams with unnecessary interventions.")
            
            if recall > 0.8:
                st.success(f"**High Recall ({recall:.1%})**: Catches most patients who will be readmitted. Good for patient safety.")
            elif recall > 0.6:
                st.warning(f"**Moderate Recall ({recall:.1%})**: Misses some readmissions. Consider additional screening methods.")
            else:
                st.error(f"**Low Recall ({recall:.1%})**: Misses many readmissions. Not suitable as primary screening tool.")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_train.columns if model_name not in ['Logistic Regression', 'Support Vector Machine', 'Neural Network'] else X_encoded.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                st.write("**📊 Most Important Factors:**")
                fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                           title=f'Top 10 Factors for {model_name}')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Explain top factors
                st.write("**🔍 Factor Explanations:**")
                factor_explanations = {
                    'age': 'Older patients have higher readmission risk due to multiple conditions and slower recovery',
                    'comorbidity_count': 'More medical conditions increase complexity and readmission likelihood',
                    'length_of_stay': 'Longer stays may indicate sicker patients or complications',
                    'risk_score': 'Overall patient risk assessment based on multiple clinical factors',
                    'is_diabetic': 'Diabetes requires ongoing management and increases readmission risk',
                    'is_hypertensive': 'High blood pressure is associated with cardiovascular complications',
                    'severity_level': 'More severe cases have higher risk of complications and readmission'
                }
                
                for _, row in importance_df.head(5).iterrows():
                    feature_name = row['feature']
                    explanation = factor_explanations.get(feature_name, 'This factor contributes to readmission risk prediction')
                    st.write(f"• **{feature_name}**: {explanation}")
        
        self.models['readmission'] = results
        return results
    
    def create_patient_segmentation(self, patients_df, encounters_df, claims_df):
        """Create patient segments using clustering algorithms"""
        
        st.info("""
        **🎯 Patient Segmentation Analysis**
        
        **What we're doing**: Grouping patients with similar characteristics to enable targeted care approaches
        
        **Why this matters**: 
        - Different patient groups need different care strategies
        - Helps allocate resources more effectively
        - Enables personalized medicine approaches
        
        **How we use the results**:
        - Design targeted care programs for each segment
        - Allocate resources based on segment needs
        - Develop personalized treatment protocols
        """)
        
        # Prepare segmentation features
        segmentation_data = patients_df.copy()
        
        # Add encounter-based features
        encounter_summary = encounters_df.groupby('patient_id').agg({
            'encounter_id': 'count',
            'length_of_stay': 'mean',
            'is_readmission': 'sum'
        }).rename(columns={
            'encounter_id': 'total_encounters',
            'length_of_stay': 'avg_length_of_stay',
            'is_readmission': 'total_readmissions'
        })
        
        # Add claims-based features
        if 'claim_amount' in claims_df.columns:
            claims_summary = claims_df.groupby('patient_id').agg({
                'claim_amount': ['sum', 'mean', 'count'],
                'paid_amount': 'sum'
            })
            claims_summary.columns = ['total_claims_amount', 'avg_claim_amount', 'total_claims', 'total_paid_amount']
        else:
            claims_summary = claims_df.groupby('patient_id').size().to_frame('total_claims')
        
        # Merge all features
        segmentation_features = segmentation_data.merge(encounter_summary, on='patient_id', how='left')
        segmentation_features = segmentation_features.merge(claims_summary, on='patient_id', how='left')
        
        # Fill missing values
        segmentation_features = segmentation_features.fillna(0)
        
        # Select features for clustering
        clustering_features = [
            'age', 'comorbidity_count', 'risk_score',
            'total_encounters', 'avg_length_of_stay', 'total_readmissions'
        ]
        
        if 'total_claims_amount' in segmentation_features.columns:
            clustering_features.extend(['total_claims_amount', 'avg_claim_amount'])
        
        X_cluster = segmentation_features[clustering_features].copy()
        
        # Scale features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Apply multiple clustering algorithms
        clustering_results = {}
        
        # K-Means Clustering
        st.write("### 🎯 K-Means Clustering")
        st.write("""
        **K-Means** groups patients by finding centers that minimize the distance to all patients in each group.
        
        **How it works**: Like organizing patients into groups where everyone in a group is as similar as possible 
        to the group's "average" patient.
        
        **Best for**: Creating balanced groups with clear separation between different patient types.
        """)
        
        # Find optimal number of clusters
        inertias = []
        silhouette_scores = []
        k_range = range(2, 8)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        
        # Choose optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Final K-means with optimal k
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_labels = kmeans_final.fit_predict(X_scaled)
        
        clustering_results['K-Means'] = {
            'labels': kmeans_labels,
            'n_clusters': optimal_k,
            'silhouette_score': silhouette_scores[np.argmax(silhouette_scores)]
        }
        
        st.write(f"**Optimal number of clusters**: {optimal_k}")
        st.write(f"**Silhouette score**: {silhouette_scores[np.argmax(silhouette_scores)]:.3f}")
        
        # Analyze clusters
        segmentation_features['kmeans_cluster'] = kmeans_labels
        
        cluster_analysis = segmentation_features.groupby('kmeans_cluster').agg({
            'age': 'mean',
            'comorbidity_count': 'mean',
            'risk_score': 'mean',
            'total_encounters': 'mean',
            'avg_length_of_stay': 'mean',
            'total_readmissions': 'mean',
            'patient_id': 'count'
        }).round(2)
        cluster_analysis.columns = ['Avg Age', 'Avg Comorbidities', 'Avg Risk Score', 
                                  'Avg Encounters', 'Avg LOS', 'Avg Readmissions', 'Patient Count']
        
        st.write("**📊 Cluster Characteristics:**")
        st.dataframe(cluster_analysis)
        
        # Create cluster profiles with healthcare interpretation
        st.write("**🏥 Clinical Cluster Profiles:**")
        
        for cluster_id in range(optimal_k):
            cluster_data = cluster_analysis.loc[cluster_id]
            
            # Determine cluster characteristics
            if cluster_data['Avg Age'] > 65 and cluster_data['Avg Comorbidities'] > 3:
                cluster_type = "Complex Elderly Patients"
                description = "High-need patients requiring comprehensive care coordination and frequent monitoring"
                care_strategy = "Geriatric care programs, medication management, fall prevention"
            elif cluster_data['Avg Risk Score'] > 0.7:
                cluster_type = "High-Risk Patients"
                description = "Patients with elevated risk requiring proactive intervention"
                care_strategy = "Care management programs, disease management, enhanced monitoring"
            elif cluster_data['Avg Encounters'] > 5:
                cluster_type = "High Utilizers"
                description = "Frequent healthcare users who may benefit from care coordination"
                care_strategy = "Case management, care coordination, address social determinants"
            elif cluster_data['Avg Comorbidities'] < 2 and cluster_data['Avg Age'] < 50:
                cluster_type = "Healthy Adults"
                description = "Generally healthy patients requiring preventive care focus"
                care_strategy = "Preventive care, wellness programs, health maintenance"
            else:
                cluster_type = "Moderate Risk Patients"
                description = "Patients with moderate healthcare needs"
                care_strategy = "Standard care protocols with periodic risk assessment"
            
            st.write(f"""
            **Cluster {cluster_id}: {cluster_type}**
            - **Size**: {cluster_data['Patient Count']:,.0f} patients
            - **Profile**: {description}
            - **Care Strategy**: {care_strategy}
            - **Key Metrics**: Age {cluster_data['Avg Age']:.0f}, Risk {cluster_data['Avg Risk Score']:.2f}, 
              Comorbidities {cluster_data['Avg Comorbidities']:.1f}
            """)
        
        # Visualize clusters
        if len(clustering_features) >= 2:
            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1], 
                color=kmeans_labels.astype(str),
                title='Patient Segments Visualization (PCA)',
                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        return clustering_results, segmentation_features
    
    def create_ml_dashboard(self, patients_df, encounters_df, claims_df):
        """Create comprehensive ML dashboard with all models and explanations"""
        
        st.markdown("# 🤖 Advanced Healthcare Machine Learning Analytics")
        
        st.markdown("""
        ## 🎯 **Machine Learning in Healthcare - Complete Guide**
        
        **What is Machine Learning?**
        Machine Learning (ML) is like teaching computers to recognize patterns in data, similar to how experienced doctors 
        develop clinical intuition over years of practice. Instead of programming specific rules, we show the computer 
        thousands of examples and let it learn the patterns.
        
        **Why Use ML in Healthcare?**
        - **Predict Outcomes**: Identify patients at risk before problems occur
        - **Optimize Resources**: Better planning for beds, staff, and equipment
        - **Personalize Care**: Tailor treatments to individual patient characteristics
        - **Improve Quality**: Reduce readmissions, complications, and medical errors
        - **Control Costs**: Focus expensive interventions on patients who need them most
        
        **How to Interpret Results:**
        - **High Accuracy**: Model makes correct predictions most of the time
        - **High Precision**: When model says "high risk," it's usually right
        - **High Recall**: Model catches most of the actual high-risk patients
        - **Feature Importance**: Shows which patient factors matter most
        """)
        
        # Model selection
        ml_options = st.multiselect(
            "Select Machine Learning Models to Run:",
            ["Readmission Prediction", "Patient Segmentation"],
            default=["Readmission Prediction"]
        )
        
        if "Readmission Prediction" in ml_options:
            st.markdown("---")
            st.markdown("## 🔄 **30-Day Readmission Prediction Models**")
            
            # Prepare and train readmission models
            model_data, features = self.prepare_readmission_data(patients_df, encounters_df, claims_df)
            readmission_results = self.train_readmission_models(model_data, features)
            
            # Model comparison
            st.markdown("### 📊 **Model Performance Comparison**")
            
            comparison_data = []
            for model_name, results in readmission_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1 Score': results['f1'],
                    'AUC Score': results['auc']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.round(3))
            
            # Best model recommendation
            best_model = comparison_df.loc[comparison_df['AUC Score'].idxmax()]
            st.success(f"""
            **🏆 Recommended Model: {best_model['Model']}**
            
            This model has the highest AUC score ({best_model['AUC Score']:.3f}), meaning it's best at 
            distinguishing between patients who will and won't be readmitted.
            
            **Clinical Application**: Use this model to identify high-risk patients during discharge planning.
            """)
        
        if "Patient Segmentation" in ml_options:
            st.markdown("---")
            st.markdown("## 👥 **Patient Segmentation Analysis**")
            
            clustering_results, segmentation_features = self.create_patient_segmentation(
                patients_df, encounters_df, claims_df
            )
            
            # Segmentation insights
            st.markdown("### 💡 **Segmentation Insights & Recommendations**")
            
            segment_counts = segmentation_features['kmeans_cluster'].value_counts().sort_index()
            
            for cluster_id in segment_counts.index:
                cluster_size = segment_counts[cluster_id]
                cluster_pct = (cluster_size / len(segmentation_features)) * 100
                
                st.write(f"""
                **Segment {cluster_id}**: {cluster_size:,} patients ({cluster_pct:.1f}% of population)
                
                **Recommended Actions**:
                - Develop targeted care protocols for this segment
                - Allocate appropriate resources based on segment characteristics
                - Monitor segment-specific quality metrics
                - Design personalized patient engagement strategies
                """)
        
        # ML Implementation Guide
        st.markdown("---")
        st.markdown("## 🚀 **Implementation Guide**")
        
        st.markdown("""
        ### **Step 1: Choose Your Model**
        - Start with the highest-performing model for your use case
        - Consider both accuracy and interpretability needs
        - Test with a small pilot group before full deployment
        
        ### **Step 2: Integrate into Workflow**
        - **Readmission Models**: Run at discharge, flag high-risk patients for care coordination
        - **Segmentation**: Update monthly, use for population health program design
        
        ### **Step 3: Monitor Performance**
        - Track model accuracy over time
        - Retrain models quarterly with new data
        - Monitor for bias across different patient populations
        - Collect feedback from clinical staff on model usefulness
        
        ### **Step 4: Continuous Improvement**
        - Add new features as data becomes available
        - Experiment with new algorithms
        - Incorporate clinical feedback into model refinements
        - Expand to additional use cases based on success
        """)
        
        return {
            'readmission_results': readmission_results if "Readmission Prediction" in ml_options else None,
            'segmentation_results': clustering_results if "Patient Segmentation" in ml_options else None
        }

# Global ML pipeline instance
comprehensive_ml_pipeline = HealthcareMLPipeline()