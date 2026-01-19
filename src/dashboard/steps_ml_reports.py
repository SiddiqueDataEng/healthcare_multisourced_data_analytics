"""
Dashboard steps for ML and reports (steps 6-7)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json

from src.ml.ml_pipeline import MLPipeline


def show_step6_machine_learning():
    """Step 6: Machine Learning"""
    st.header("Step 6: Machine Learning Models")
    st.markdown("Train and evaluate predictive models")
    
    patients = st.session_state.patients_df
    encounters = st.session_state.encounters_df
    claims = st.session_state.claims_df
    
    ml_pipeline = MLPipeline()
    
    # Model Training Section
    st.subheader("🤖 Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_readmission = st.checkbox("Train Readmission Prediction", value=True)
    with col2:
        train_cost = st.checkbox("Train Cost Prediction", value=True)
    
    if st.button("🚀 Train Models", type="primary"):
        results = {}
        
        # Readmission Prediction
        if train_readmission:
            with st.spinner("Training readmission prediction models..."):
                features = ml_pipeline.engineer_readmission_features(
                    patients, encounters, claims
                )
                readmission_results = ml_pipeline.train_readmission_models(features)
                results['readmission'] = readmission_results
                st.success("✅ Readmission models trained!")
        
        # Cost Prediction
        if train_cost:
            with st.spinner("Training cost prediction models..."):
                cost_features = ml_pipeline.engineer_cost_features(
                    patients, encounters, claims
                )
                cost_results = ml_pipeline.train_cost_prediction_models(cost_features)
                results['cost'] = cost_results
                st.success("✅ Cost models trained!")
        
        st.session_state.ml_results = results
        st.rerun()
    
    st.divider()
    
    # Display Results
    if st.session_state.ml_results:
        results = st.session_state.ml_results
        
        # Readmission Results
        if 'readmission' in results:
            st.subheader("🔮 Readmission Prediction Results")
            
            r = results['readmission']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Model", r['best_model'])
            with col2:
                st.metric("ROC-AUC", f"{r['best_roc_auc']:.3f}")
            with col3:
                st.metric("Train Size", f"{r['train_size']:,}")
            
            # Model comparison
            st.markdown("**Model Comparison**")
            
            model_comparison = pd.DataFrame([
                {
                    'Model': name,
                    'Accuracy': f"{metrics['accuracy']:.3f}",
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'F1': f"{metrics['f1_score']:.3f}",
                    'ROC-AUC': f"{metrics['roc_auc']:.3f}"
                }
                for name, metrics in r['models'].items()
            ])
            
            st.dataframe(model_comparison, use_container_width=True, hide_index=True)
            
            # Feature importance
            best_model_metrics = r['models'][r['best_model']]
            if 'feature_importance' in best_model_metrics:
                st.markdown("**Top 10 Features**")
                
                features_df = pd.DataFrame(best_model_metrics['feature_importance'][:10])
                
                fig = px.bar(features_df, x='importance', y='feature',
                           orientation='h',
                           title='Feature Importance',
                           labels={'importance': 'Importance', 'feature': 'Feature'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
        
        # Cost Prediction Results
        if 'cost' in results:
            st.subheader("💰 Cost Prediction Results")
            
            c = results['cost']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Model", c['best_model'])
            with col2:
                st.metric("R² Score", f"{c['best_r2_score']:.3f}")
            with col3:
                st.metric("Train Size", f"{c['train_size']:,}")
            
            # Model comparison
            st.markdown("**Model Comparison**")
            
            model_comparison = pd.DataFrame([
                {
                    'Model': name,
                    'RMSE': f"${metrics['rmse']:,.2f}",
                    'MAE': f"${metrics['mae']:,.2f}",
                    'R²': f"{metrics['r2_score']:.3f}",
                    'Mean Actual': f"${metrics['mean_actual']:,.2f}",
                    'Mean Predicted': f"${metrics['mean_predicted']:,.2f}"
                }
                for name, metrics in c['models'].items()
            ])
            
            st.dataframe(model_comparison, use_container_width=True, hide_index=True)
            
            # Feature importance
            best_model_metrics = c['models'][c['best_model']]
            if 'feature_importance' in best_model_metrics:
                st.markdown("**Top Features**")
                
                features_df = pd.DataFrame(best_model_metrics['feature_importance'])
                
                fig = px.bar(features_df, x='importance', y='feature',
                           orientation='h',
                           title='Feature Importance',
                           labels={'importance': 'Importance', 'feature': 'Feature'})
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("ℹ️ No models trained yet. Click 'Train Models' to start.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 5
            st.rerun()
    with col2:
        if st.button("Next: Reports ➡️", type="primary"):
            st.session_state.step = 7
            st.rerun()



def show_step7_reports():
    """Step 7: Reports & Export"""
    st.header("Step 7: Reports & Export")
    st.markdown("Generate and export comprehensive reports")
    
    patients = st.session_state.patients_df
    encounters = st.session_state.encounters_df
    claims = st.session_state.claims_df
    
    # Summary Report
    st.subheader("📊 Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Patients", f"{len(patients):,}")
        st.metric("Total Encounters", f"{len(encounters):,}")
        st.metric("Total Claims", f"{len(claims):,}")
    
    with col2:
        if 'age' in patients.columns:
            st.metric("Avg Age", f"{patients['age'].mean():.1f}")
        if 'comorbidity_count' in patients.columns:
            st.metric("Avg Comorbidities", f"{patients['comorbidity_count'].mean():.1f}")
        if 'length_of_stay' in encounters.columns:
            st.metric("Avg LOS", f"{encounters['length_of_stay'].mean():.1f} days")
    
    with col3:
        if 'is_readmission' in encounters.columns:
            readmission_rate = encounters['is_readmission'].mean()
            st.metric("Readmission Rate", f"{readmission_rate:.1%}")
        if 'claim_amount' in claims.columns:
            st.metric("Avg Claim", f"${claims['claim_amount'].mean():,.2f}")
    
    st.divider()
    
    # Cohort Summary
    if 'cohorts' in patients.columns:
        st.subheader("👥 Cohort Summary")
        
        cohort_counts = {}
        for cohorts_str in patients['cohorts'].dropna():
            if cohorts_str:
                for cohort in cohorts_str.split('|'):
                    cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1
        
        if cohort_counts:
            cohort_df = pd.DataFrame(list(cohort_counts.items()), 
                                    columns=['Cohort', 'Count'])
            cohort_df['Percentage'] = (cohort_df['Count'] / len(patients) * 100).round(1)
            cohort_df = cohort_df.sort_values('Count', ascending=False)
            
            st.dataframe(cohort_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # ML Results Summary
    if st.session_state.ml_results:
        st.subheader("🤖 ML Model Performance")
        
        results = st.session_state.ml_results
        
        if 'readmission' in results:
            r = results['readmission']
            st.write(f"**Readmission Prediction**: {r['best_model']} (ROC-AUC: {r['best_roc_auc']:.3f})")
        
        if 'cost' in results:
            c = results['cost']
            st.write(f"**Cost Prediction**: {c['best_model']} (R²: {c['best_r2_score']:.3f})")
    
    st.divider()
    
    # Export Options
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Summary (CSV)"):
            summary_data = {
                'Metric': ['Total Patients', 'Total Encounters', 'Total Claims',
                          'Avg Age', 'Avg Comorbidities', 'Readmission Rate'],
                'Value': [
                    len(patients),
                    len(encounters),
                    len(claims),
                    patients['age'].mean() if 'age' in patients.columns else 0,
                    patients['comorbidity_count'].mean() if 'comorbidity_count' in patients.columns else 0,
                    encounters['is_readmission'].mean() if 'is_readmission' in encounters.columns else 0
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="analytics_summary.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Export ML Results (JSON)"):
            if st.session_state.ml_results:
                json_str = json.dumps(st.session_state.ml_results, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="ml_results.json",
                    mime="application/json"
                )
            else:
                st.warning("No ML results to export")
    
    with col3:
        if st.button("📋 Generate Full Report"):
            st.info("Full report generation coming soon!")
    
    st.divider()
    
    # Next Steps
    st.subheader("🎯 Next Steps")
    
    st.markdown("""
    **Congratulations!** You've completed the comprehensive analytics workflow.
    
    **What you can do next:**
    
    1. **Re-run Analysis**: Click "Reset Workflow" in sidebar to start over with new data
    2. **Export Results**: Use the export buttons above to save your findings
    3. **Explore More**: Use the quick navigation to jump to specific analyses
    4. **Generate New Data**: Run `generate_data_medium.bat` to create new datasets
    5. **View Detailed Reports**: Check `data/curated/analytics/` for JSON results
    
    **Batch Scripts Available:**
    - `run_analytics_only.bat` - Run analytics without UI
    - `run_ml_only.bat` - Train ML models only
    - `view_results.bat` - View all generated reports
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 6
            st.rerun()
    with col2:
        if st.button("🔄 Start Over", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
