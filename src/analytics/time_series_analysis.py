"""
Time Series Analysis for Healthcare Data
Includes trend analysis, seasonal patterns, and forecasting
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """Analyze temporal patterns and forecast future trends"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def encounter_trends(self, encounters_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze encounter volume trends over time
        
        Returns:
            - Daily/monthly encounter counts
            - Trend direction
            - Growth rate
            - Seasonal patterns
        """
        self.logger.info("Analyzing encounter trends")
        
        # Ensure date column is datetime
        encounters_df = encounters_df.copy()
        encounters_df['encounter_date'] = pd.to_datetime(encounters_df['encounter_date'])
        
        # Daily counts
        daily_counts = encounters_df.groupby('encounter_date').size().reset_index(name='count')
        daily_counts = daily_counts.sort_values('encounter_date')
        
        # Monthly counts
        encounters_df['year_month'] = encounters_df['encounter_date'].dt.to_period('M')
        monthly_counts = encounters_df.groupby('year_month').size().reset_index(name='count')
        monthly_counts['year_month'] = monthly_counts['year_month'].astype(str)
        
        # Calculate trend
        daily_counts['day_num'] = (daily_counts['encounter_date'] - daily_counts['encounter_date'].min()).dt.days
        trend_slope = np.polyfit(daily_counts['day_num'], daily_counts['count'], 1)[0]
        
        # Calculate growth rate
        first_month = monthly_counts['count'].iloc[0]
        last_month = monthly_counts['count'].iloc[-1]
        num_months = len(monthly_counts)
        monthly_growth_rate = ((last_month / first_month) ** (1 / num_months) - 1) * 100
        
        result = {
            'total_encounters': int(len(encounters_df)),
            'date_range': {
                'start': str(encounters_df['encounter_date'].min().date()),
                'end': str(encounters_df['encounter_date'].max().date()),
                'days': int((encounters_df['encounter_date'].max() - encounters_df['encounter_date'].min()).days)
            },
            
            # Daily statistics
            'daily_avg': float(daily_counts['count'].mean()),
            'daily_median': float(daily_counts['count'].median()),
            'daily_std': float(daily_counts['count'].std()),
            'daily_min': int(daily_counts['count'].min()),
            'daily_max': int(daily_counts['count'].max()),
            
            # Monthly statistics
            'monthly_avg': float(monthly_counts['count'].mean()),
            'monthly_median': float(monthly_counts['count'].median()),
            
            # Trend
            'trend_slope': float(trend_slope),
            'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
            'monthly_growth_rate_pct': float(monthly_growth_rate),
            
            # Time series data
            'daily_series': daily_counts[['encounter_date', 'count']].to_dict('records'),
            'monthly_series': monthly_counts.to_dict('records'),
        }
        
        self.logger.info(f"Trend: {result['trend_direction']}, growth rate: {monthly_growth_rate:.2f}%/month")
        return result
    
    def seasonal_patterns(self, encounters_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify seasonal patterns in encounters
        
        Returns:
            - Monthly patterns
            - Day of week patterns
            - Seasonal indices
        """
        self.logger.info("Analyzing seasonal patterns")
        
        encounters_df = encounters_df.copy()
        encounters_df['encounter_date'] = pd.to_datetime(encounters_df['encounter_date'])
        
        # Extract temporal features
        encounters_df['month'] = encounters_df['encounter_date'].dt.month
        encounters_df['month_name'] = encounters_df['encounter_date'].dt.month_name()
        encounters_df['day_of_week'] = encounters_df['encounter_date'].dt.day_name()
        encounters_df['quarter'] = encounters_df['encounter_date'].dt.quarter
        
        # Monthly pattern
        monthly_pattern = encounters_df.groupby('month_name').size()
        monthly_avg = monthly_pattern.mean()
        monthly_index = (monthly_pattern / monthly_avg * 100).round(1)
        
        # Day of week pattern
        dow_pattern = encounters_df.groupby('day_of_week').size()
        dow_avg = dow_pattern.mean()
        dow_index = (dow_pattern / dow_avg * 100).round(1)
        
        # Quarterly pattern
        quarterly_pattern = encounters_df.groupby('quarter').size()
        
        # Identify peak and low periods
        peak_month = monthly_pattern.idxmax()
        low_month = monthly_pattern.idxmin()
        peak_dow = dow_pattern.idxmax()
        low_dow = dow_pattern.idxmin()
        
        result = {
            # Monthly patterns
            'monthly_counts': monthly_pattern.to_dict(),
            'monthly_index': monthly_index.to_dict(),
            'peak_month': peak_month,
            'low_month': low_month,
            'monthly_variation_pct': float((monthly_pattern.max() - monthly_pattern.min()) / monthly_avg * 100),
            
            # Day of week patterns
            'day_of_week_counts': dow_pattern.to_dict(),
            'day_of_week_index': dow_index.to_dict(),
            'peak_day': peak_dow,
            'low_day': low_dow,
            
            # Quarterly patterns
            'quarterly_counts': quarterly_pattern.to_dict(),
            
            # Interpretation
            'has_winter_surge': monthly_index.get('December', 100) > 110 or monthly_index.get('January', 100) > 110,
            'has_weekday_pattern': dow_index.get('Saturday', 100) < 90 or dow_index.get('Sunday', 100) < 90,
        }
        
        self.logger.info(f"Peak month: {peak_month}, Low month: {low_month}")
        return result
    
    def cost_trends(self, claims_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze cost trends over time
        
        Returns:
            - Monthly cost trends
            - Cost growth rate
            - Cost per claim trends
        """
        self.logger.info("Analyzing cost trends")
        
        claims_df = claims_df.copy()
        claims_df['claim_date'] = pd.to_datetime(claims_df['claim_date'])
        claims_df['year_month'] = claims_df['claim_date'].dt.to_period('M')
        
        # Monthly aggregations
        monthly_costs = claims_df.groupby('year_month').agg({
            'claim_amount': ['sum', 'mean', 'count'],
            'paid_amount': ['sum', 'mean']
        }).reset_index()
        
        monthly_costs.columns = ['year_month', 'total_claim', 'avg_claim', 'claim_count', 
                                 'total_paid', 'avg_paid']
        monthly_costs['year_month'] = monthly_costs['year_month'].astype(str)
        
        # Calculate growth rates
        first_month_cost = monthly_costs['total_claim'].iloc[0]
        last_month_cost = monthly_costs['total_claim'].iloc[-1]
        num_months = len(monthly_costs)
        cost_growth_rate = ((last_month_cost / first_month_cost) ** (1 / num_months) - 1) * 100
        
        result = {
            'total_claims': int(len(claims_df)),
            'total_claim_amount': float(claims_df['claim_amount'].sum()),
            'total_paid_amount': float(claims_df['paid_amount'].sum()),
            
            # Monthly trends
            'monthly_avg_total_cost': float(monthly_costs['total_claim'].mean()),
            'monthly_avg_claim_count': float(monthly_costs['claim_count'].mean()),
            'monthly_cost_growth_rate_pct': float(cost_growth_rate),
            
            # Cost per claim trends
            'avg_cost_per_claim': float(claims_df['claim_amount'].mean()),
            'avg_paid_per_claim': float(claims_df['paid_amount'].mean()),
            
            # Time series
            'monthly_series': monthly_costs.to_dict('records'),
        }
        
        self.logger.info(f"Cost growth rate: {cost_growth_rate:.2f}%/month")
        return result
    
    def disease_prevalence_trends(self, encounters_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze disease prevalence trends over time
        
        Returns:
            - Top diagnoses by time period
            - Emerging conditions
            - Declining conditions
        """
        self.logger.info("Analyzing disease prevalence trends")
        
        encounters_df = encounters_df.copy()
        encounters_df['encounter_date'] = pd.to_datetime(encounters_df['encounter_date'])
        encounters_df['year_month'] = encounters_df['encounter_date'].dt.to_period('M')
        
        # Get top diagnoses
        top_diagnoses = encounters_df['primary_diagnosis'].value_counts().head(10).index.tolist()
        
        # Track each diagnosis over time
        diagnosis_trends = {}
        for diagnosis in top_diagnoses:
            monthly_counts = encounters_df[encounters_df['primary_diagnosis'] == diagnosis].groupby('year_month').size()
            diagnosis_trends[diagnosis] = {
                'total_count': int(monthly_counts.sum()),
                'monthly_avg': float(monthly_counts.mean()),
                'trend': 'increasing' if monthly_counts.iloc[-3:].mean() > monthly_counts.iloc[:3].mean() else 'decreasing'
            }
        
        result = {
            'top_diagnoses': list(top_diagnoses),
            'diagnosis_trends': diagnosis_trends,
        }
        
        return result
    
    def forecast_encounters(self, encounters_df: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
        """
        Forecast future encounter volumes using simple moving average and trend
        
        Args:
            periods: Number of days to forecast
        
        Returns:
            - Forecasted values
            - Confidence intervals
            - Forecast method
        """
        self.logger.info(f"Forecasting encounters for next {periods} days")
        
        encounters_df = encounters_df.copy()
        encounters_df['encounter_date'] = pd.to_datetime(encounters_df['encounter_date'])
        
        # Daily counts
        daily_counts = encounters_df.groupby('encounter_date').size().reset_index(name='count')
        daily_counts = daily_counts.sort_values('encounter_date')
        
        # Calculate trend using linear regression
        daily_counts['day_num'] = (daily_counts['encounter_date'] - daily_counts['encounter_date'].min()).dt.days
        trend_coef = np.polyfit(daily_counts['day_num'], daily_counts['count'], 1)
        
        # Calculate moving average
        window = min(30, len(daily_counts) // 2)
        daily_counts['ma'] = daily_counts['count'].rolling(window=window, min_periods=1).mean()
        
        # Generate forecast dates
        last_date = daily_counts['encounter_date'].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        # Calculate forecast
        last_day_num = daily_counts['day_num'].max()
        forecast_day_nums = np.arange(last_day_num + 1, last_day_num + periods + 1)
        
        # Trend-based forecast
        forecast_values = trend_coef[0] * forecast_day_nums + trend_coef[1]
        
        # Add seasonal adjustment if available
        if len(daily_counts) >= 365:
            # Simple seasonal adjustment based on day of year
            daily_counts['day_of_year'] = daily_counts['encounter_date'].dt.dayofyear
            seasonal_avg = daily_counts.groupby('day_of_year')['count'].mean()
            
            forecast_doy = [d.dayofyear for d in forecast_dates]
            seasonal_factors = [seasonal_avg.get(doy, 1.0) / daily_counts['count'].mean() for doy in forecast_doy]
            forecast_values = forecast_values * seasonal_factors
        
        # Calculate confidence intervals (simple approach using historical std)
        std = daily_counts['count'].std()
        lower_bound = forecast_values - 1.96 * std
        upper_bound = forecast_values + 1.96 * std
        
        # Ensure non-negative
        forecast_values = np.maximum(forecast_values, 0)
        lower_bound = np.maximum(lower_bound, 0)
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_values,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
        
        result = {
            'forecast_method': 'Linear trend with seasonal adjustment',
            'forecast_periods': periods,
            'forecast_start_date': str(forecast_dates[0].date()),
            'forecast_end_date': str(forecast_dates[-1].date()),
            
            # Historical statistics
            'historical_avg': float(daily_counts['count'].mean()),
            'historical_std': float(std),
            'historical_trend_slope': float(trend_coef[0]),
            
            # Forecast summary
            'forecast_avg': float(forecast_values.mean()),
            'forecast_total': float(forecast_values.sum()),
            
            # Forecast data
            'forecast_series': forecast_df.to_dict('records'),
        }
        
        self.logger.info(f"Forecast complete: avg {forecast_values.mean():.1f} encounters/day")
        return result
    
    def forecast_costs(self, claims_df: pd.DataFrame, periods: int = 12) -> Dict[str, Any]:
        """
        Forecast future costs (monthly)
        
        Args:
            periods: Number of months to forecast
        
        Returns:
            - Forecasted monthly costs
            - Confidence intervals
        """
        self.logger.info(f"Forecasting costs for next {periods} months")
        
        claims_df = claims_df.copy()
        claims_df['claim_date'] = pd.to_datetime(claims_df['claim_date'])
        claims_df['year_month'] = claims_df['claim_date'].dt.to_period('M')
        
        # Monthly costs
        monthly_costs = claims_df.groupby('year_month')['claim_amount'].sum().reset_index()
        monthly_costs['month_num'] = range(len(monthly_costs))
        
        # Fit trend
        trend_coef = np.polyfit(monthly_costs['month_num'], monthly_costs['claim_amount'], 1)
        
        # Generate forecast
        last_month_num = monthly_costs['month_num'].max()
        forecast_month_nums = np.arange(last_month_num + 1, last_month_num + periods + 1)
        forecast_values = trend_coef[0] * forecast_month_nums + trend_coef[1]
        
        # Confidence intervals
        std = monthly_costs['claim_amount'].std()
        lower_bound = forecast_values - 1.96 * std
        upper_bound = forecast_values + 1.96 * std
        
        # Ensure non-negative
        forecast_values = np.maximum(forecast_values, 0)
        lower_bound = np.maximum(lower_bound, 0)
        
        result = {
            'forecast_method': 'Linear trend',
            'forecast_periods': periods,
            
            # Historical
            'historical_monthly_avg': float(monthly_costs['claim_amount'].mean()),
            'historical_monthly_std': float(std),
            'monthly_growth_rate': float(trend_coef[0]),
            
            # Forecast
            'forecast_monthly_avg': float(forecast_values.mean()),
            'forecast_total': float(forecast_values.sum()),
            
            # Forecast data
            'forecast_values': forecast_values.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
        }
        
        self.logger.info(f"Forecast complete: avg ${forecast_values.mean():,.0f}/month")
        return result
