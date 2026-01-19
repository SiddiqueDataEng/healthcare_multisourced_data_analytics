"""
Dashboard steps for analytics (steps 3-5)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.statistical_analysis import StatisticalAnalyzer
from src.analytics.cohort_analysis import CohortAnalyzer
from src.analytics.time_series_analysis import TimeSeriesAnalyzer


def show_step3_statistical_analysis():
    """Step 3: Statistical Analysis"""
    st.header("Step 3: Statistical Analysis")
    st.markdown("Comprehensive statistical analysis of your data")
    
    patients = st.session_state.patients_df
    encounters = st.session_state.encounters_df
    claims = st.session_state.claims_df
    
    analyzer = StatisticalAnalyzer()
    
    # Descriptive Statistics
    st.subheader("📊 Descriptive Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Patient Age Statistics**")
        if 'age' in patients.columns:
            age_stats = analyzer.descriptive_statistics(patients, ['age'])
            if 'age' in age_stats:
                stats = age_stats['age']
                st.write(f"- Mean: {stats['mean']:.1f} years")
                st.write(f"- Median: {stats['median']:.1f} years")
                st.write(f"- Std Dev: {stats['std']:.1f}")
                st.write(f"- Range: {stats['min']:.0f} - {stats['max']:.0f}")
                st.write(f"- Skewness: {stats['skewness']:.2f}")
    
    with col2:
        st.markdown("**Length of Stay Statistics**")
        if 'length_of_stay' in encounters.columns:
            los_stats = analyzer.descriptive_statistics(encounters, ['length_of_stay'])
            if 'length_of_stay' in los_stats:
                stats = los_stats['length_of_stay']
                st.write(f"- Mean: {stats['mean']:.1f} days")
                st.write(f"- Median: {stats['median']:.1f} days")
                st.write(f"- Std Dev: {stats['std']:.1f}")
                st.write(f"- Range: {stats['min']:.0f} - {stats['max']:.0f}")
    
    st.divider()
    
    # Hypothesis Testing
    st.subheader("🔬 Hypothesis Testing")
    
    if 'is_diabetic' in patients.columns and 'length_of_stay' in encounters.columns:
        test_df = encounters.merge(patients[['patient_id', 'is_diabetic']], on='patient_id')
        
        with st.spinner("Running hypothesis test..."):
            result = analyzer.hypothesis_testing(test_df, 'is_diabetic', 'length_of_stay')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Diabetic Mean LOS", f"{result['group1_mean']:.1f} days")
            with col2:
                st.metric("Non-Diabetic Mean LOS", f"{result['group2_mean']:.1f} days")
            with col3:
                st.metric("P-Value", f"{result['pvalue']:.4f}")
            
            if result['significant']:
                st.success(f"✅ {result['interpretation']}")
            else:
                st.info(f"ℹ️ {result['interpretation']}")
    
    st.divider()
    
    # Correlation Analysis
    st.subheader("🔗 Correlation Analysis")
    
    if 'age' in patients.columns and 'comorbidity_count' in patients.columns:
        with st.spinner("Calculating correlations..."):
            corr_result = analyzer.correlation_analysis(
                patients, ['age', 'comorbidity_count', 'risk_score'], method='pearson'
            )
            
            if corr_result['significant_pairs']:
                st.write(f"Found {len(corr_result['significant_pairs'])} significant correlations:")
                for pair in corr_result['significant_pairs'][:5]:
                    st.write(f"- **{pair['var1']}** ↔ **{pair['var2']}**: "
                           f"r = {pair['correlation']:.3f} ({pair['strength']})")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Next: Cohort Analysis ➡️", type="primary"):
            st.session_state.step = 4
            st.rerun()



def show_step4_cohort_analysis():
    """Step 4: Cohort Analysis"""
    st.header("Step 4: Cohort Analysis")
    st.markdown("Analyze patient cohorts and their outcomes")
    
    patients = st.session_state.patients_df
    encounters = st.session_state.encounters_df
    claims = st.session_state.claims_df
    
    analyzer = CohortAnalyzer()
    
    # Identify cohorts
    with st.spinner("Identifying cohorts..."):
        cohorts = analyzer.identify_cohorts(patients)
    
    st.subheader("📊 Identified Cohorts")
    
    cohort_summary = pd.DataFrame([
        {
            'Cohort': name.title(),
            'Patients': len(df),
            'Percentage': f"{len(df)/len(patients)*100:.1f}%"
        }
        for name, df in cohorts.items()
    ])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(cohort_summary, x='Cohort', y='Patients',
                   title='Cohort Sizes',
                   color='Patients', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(cohort_summary, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Select cohort for detailed analysis
    st.subheader("🔍 Detailed Cohort Analysis")
    
    cohort_names = list(cohorts.keys())
    selected_cohort = st.selectbox("Select a cohort to analyze:", cohort_names)
    
    if selected_cohort:
        cohort_df = cohorts[selected_cohort]
        
        # Demographics
        with st.spinner(f"Analyzing {selected_cohort} cohort..."):
            demographics = analyzer.cohort_demographics(cohort_df, selected_cohort)
            outcomes = analyzer.cohort_outcomes(cohort_df, encounters, claims, selected_cohort)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Age", f"{demographics['age_mean']:.1f}")
        with col2:
            st.metric("Avg Comorbidities", f"{demographics['avg_comorbidities']:.1f}")
        with col3:
            st.metric("Encounters/Patient", f"{outcomes['encounters_per_patient']:.1f}")
        with col4:
            if outcomes['readmission_rate']:
                st.metric("Readmission Rate", f"{outcomes['readmission_rate']:.1%}")
        
        # Outcomes
        st.markdown("**Clinical & Financial Outcomes**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if outcomes['avg_length_of_stay']:
                st.write(f"- Avg Length of Stay: {outcomes['avg_length_of_stay']:.1f} days")
            st.write(f"- Total Encounters: {outcomes['total_encounters']:,}")
            st.write(f"- Total Claims: {outcomes['total_claims']:,}")
        
        with col2:
            if outcomes['avg_claim_per_patient']:
                st.write(f"- Avg Cost/Patient: ${outcomes['avg_claim_per_patient']:,.2f}")
            if outcomes['total_claim_amount']:
                st.write(f"- Total Claims Amount: ${outcomes['total_claim_amount']:,.2f}")
    
    st.divider()
    
    # Cohort Comparison
    st.subheader("⚖️ Compare Cohorts")
    
    col1, col2 = st.columns(2)
    with col1:
        cohort1 = st.selectbox("Cohort 1:", cohort_names, key='cohort1')
    with col2:
        cohort2 = st.selectbox("Cohort 2:", [c for c in cohort_names if c != cohort1], key='cohort2')
    
    if st.button("Compare Cohorts"):
        with st.spinner("Comparing cohorts..."):
            comparison = analyzer.compare_cohorts(
                cohorts[cohort1], cohorts[cohort2],
                encounters, claims,
                cohort1, cohort2
            )
            
            st.write(f"**{cohort1.title()} vs {cohort2.title()}**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Size Difference", 
                         f"{comparison['cohort1_size'] - comparison['cohort2_size']:,}")
            
            with col2:
                if comparison['encounters_per_patient_pct_diff']:
                    st.metric("Encounters Diff", 
                             f"{comparison['encounters_per_patient_pct_diff']:.1f}%")
            
            with col3:
                if comparison['avg_cost_per_patient_pct_diff']:
                    st.metric("Cost Diff", 
                             f"{comparison['avg_cost_per_patient_pct_diff']:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("Next: Time Series ➡️", type="primary"):
            st.session_state.step = 5
            st.rerun()



def show_step5_time_series():
    """Step 5: Time Series & Forecasting"""
    st.header("Step 5: Time Series Analysis & Forecasting")
    st.markdown("Analyze trends and forecast future patterns")
    
    encounters = st.session_state.encounters_df
    claims = st.session_state.claims_df
    
    analyzer = TimeSeriesAnalyzer()
    
    # Encounter Trends
    st.subheader("📈 Encounter Trends")
    
    with st.spinner("Analyzing encounter trends..."):
        trends = analyzer.encounter_trends(encounters)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Daily Average", f"{trends['daily_avg']:.1f}")
    with col2:
        st.metric("Trend", trends['trend_direction'].title())
    with col3:
        st.metric("Growth Rate", f"{trends['monthly_growth_rate_pct']:.2f}%/mo")
    with col4:
        st.metric("Total Days", trends['date_range']['days'])
    
    # Plot daily trend
    if 'daily_series' in trends:
        daily_df = pd.DataFrame(trends['daily_series'])
        daily_df['encounter_date'] = pd.to_datetime(daily_df['encounter_date'])
        
        fig = px.line(daily_df, x='encounter_date', y='count',
                     title='Daily Encounter Volume',
                     labels={'encounter_date': 'Date', 'count': 'Encounters'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Seasonal Patterns
    st.subheader("🌡️ Seasonal Patterns")
    
    with st.spinner("Analyzing seasonal patterns..."):
        patterns = analyzer.seasonal_patterns(encounters)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Peak Month**: {patterns['peak_month']}")
        st.write(f"**Low Month**: {patterns['low_month']}")
        st.write(f"**Winter Surge**: {'Yes' if patterns['has_winter_surge'] else 'No'}")
    
    with col2:
        st.write(f"**Peak Day**: {patterns['peak_day']}")
        st.write(f"**Low Day**: {patterns['low_day']}")
        st.write(f"**Weekday Pattern**: {'Yes' if patterns['has_weekday_pattern'] else 'No'}")
    
    # Monthly pattern chart
    if 'monthly_counts' in patterns:
        monthly_df = pd.DataFrame(list(patterns['monthly_counts'].items()),
                                 columns=['Month', 'Count'])
        
        fig = px.bar(monthly_df, x='Month', y='Count',
                    title='Encounters by Month',
                    color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Forecasting
    st.subheader("🔮 Forecasting")
    
    forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
    
    if st.button("Generate Forecast"):
        with st.spinner(f"Forecasting next {forecast_days} days..."):
            forecast = analyzer.forecast_encounters(encounters, periods=forecast_days)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Forecast Avg", f"{forecast['forecast_avg']:.1f}/day")
            with col2:
                st.metric("Forecast Total", f"{forecast['forecast_total']:.0f}")
            with col3:
                st.metric("Historical Avg", f"{forecast['historical_avg']:.1f}/day")
            
            # Plot forecast
            forecast_df = pd.DataFrame(forecast['forecast_series'])
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['upper_bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['lower_bound'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fillcolor='rgba(0,100,200,0.2)',
                fill='tonexty',
                showlegend=False
            ))
            
            fig.update_layout(
                title=f'{forecast_days}-Day Encounter Forecast',
                xaxis_title='Date',
                yaxis_title='Encounters',
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("Next: Machine Learning ➡️", type="primary"):
            st.session_state.step = 6
            st.rerun()
