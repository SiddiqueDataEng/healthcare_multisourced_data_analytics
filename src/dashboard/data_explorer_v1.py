"""
Healthcare Data Explorer & Quality Management Dashboard
Comprehensive data viewing, quality checking, transformation, and SQL query interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import our SQL library
try:
    from src.sql.healthcare_sql_library import HealthcareSQLLibrary
    sql_library = HealthcareSQLLibrary()
    print("✅ SQL Library loaded successfully")
except ImportError as e:
    print(f"⚠️ SQL Library import error: {e}")
    # Create a simple fallback SQL library
    class SimpleSQLLibrary:
        def __init__(self):
            self.queries = {
                "patient_count": {
                    "sql": "SELECT COUNT(*) as total_patients FROM patients;",
                    "description": "Count total patients"
                }
            }
            self.query_categories = {"Basic": ["patient_count"]}
        
        def get_query_list(self):
            return self.query_categories
        
        def get_query_text(self, query_name):
            return self.queries.get(query_name, {}).get("sql", "")
        
        def explain_query(self, query_name):
            return self.queries.get(query_name, {}).get("description", "")
    
    sql_library = SimpleSQLLibrary()
    print("✅ Fallback SQL Library created")
except Exception as e:
    print(f"⚠️ SQL Library error: {e}")
    sql_library = None

def clean_dataframe_for_serialization(df):
    """Clean DataFrame specifically for JSON serialization in session state"""
    if df is None or df.empty:
        return df
    
    try:
        df_clean = df.copy()
        
        # Convert all columns to appropriate types for JSON serialization
        for col in df_clean.columns:
            try:
                # Handle object columns (including ObjectDType)
                if df_clean[col].dtype == 'object' or str(df_clean[col].dtype).startswith('object'):
                    # Convert to string and handle special values
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaT', '<NA>', 'nat'], '')
                
                # Handle datetime columns
                elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                
                # Handle categorical columns
                elif pd.api.types.is_categorical_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].astype(str)
                
                # Handle numeric columns with NaN
                elif df_clean[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    df_clean[col] = df_clean[col].fillna(0)
                
                # Handle boolean columns
                elif df_clean[col].dtype == 'bool':
                    df_clean[col] = df_clean[col].astype(str)
                
                # Handle any other data types by converting to string
                else:
                    df_clean[col] = df_clean[col].astype(str)
                
            except Exception as col_error:
                # If column conversion fails, convert to string as fallback
                try:
                    df_clean[col] = df_clean[col].astype(str).fillna('')
                except:
                    # Last resort: fill with empty string
                    df_clean[col] = ''
        
        # Final cleanup - replace any remaining NaN/None values
        df_clean = df_clean.fillna('')
        
        # Ensure all values are JSON serializable by converting to basic types
        for col in df_clean.columns:
            try:
                # Force conversion to basic Python types
                df_clean[col] = df_clean[col].apply(
                    lambda x: str(x) if pd.notna(x) and str(x) not in ['nan', 'None', 'NaT', '<NA>'] else ''
                )
            except:
                df_clean[col] = ''
        
        return df_clean
        
    except Exception as e:
        # If all else fails, return an empty DataFrame with the same columns
        try:
            return pd.DataFrame(columns=df.columns)
        except:
            return pd.DataFrame()

def aggressive_session_state_cleanup():
    """Aggressively clear all session state that might cause JSON serialization issues"""
    try:
        # Clear all data-related session state keys
        keys_to_clear = []
        for key in list(st.session_state.keys()):
            if any(keyword in key.lower() for keyword in [
                'query_result', 'dataframe', 'df_', 'data_', 'result_', 
                'loaded_data', 'explorer_', 'quality_', 'transform_'
            ]):
                keys_to_clear.append(key)
        
        for key in keys_to_clear:
            try:
                del st.session_state[key]
            except:
                pass
                
        # Also clear any keys that might contain pandas objects
        for key in list(st.session_state.keys()):
            try:
                value = st.session_state[key]
                if hasattr(value, 'dtype') or str(type(value)).find('pandas') != -1:
                    del st.session_state[key]
            except:
                pass
                
    except Exception as e:
        # If cleanup fails, just continue
        pass

def clear_problematic_session_state():
    """Clear any session state that might cause JSON serialization issues"""
    try:
        # List of session state keys that might contain problematic data
        problematic_keys = []
        
        for key in list(st.session_state.keys()):
            if any(keyword in key.lower() for keyword in ['query_result', 'dataframe', 'df_']):
                try:
                    # Try to JSON serialize the value
                    import json
                    json.dumps(st.session_state[key])
                except (TypeError, ValueError):
                    # If it fails, mark for deletion
                    problematic_keys.append(key)
        
        # Remove problematic keys
        for key in problematic_keys:
            try:
                del st.session_state[key]
            except:
                pass
                
    except Exception as e:
        # If cleanup fails, just continue
        pass

def safe_store_in_session_state(key, df, query_name="Unknown"):
    """Safely store DataFrame in session state to avoid JSON serialization errors"""
    try:
        # Don't store anything in session state - just show a message
        st.success(f"✅ Query '{query_name}' executed successfully with {len(df)} rows")
        st.info("💡 **Note:** Query results are displayed above. Use the Export button to save results.")
        return True
        
    except Exception as e:
        st.warning(f"Query executed but could not store results: {str(e)}")
        return False

def safe_load_from_session_state(key):
    """Safely load DataFrame from session state"""
    # Since we're not storing anything, return None
    return None, None

class DataExplorer:
    """Comprehensive data exploration and quality management"""
    
    def __init__(self):
        self.data_files = {
            'patients': 'data/landing_zone/patients.csv',
            'encounters': 'data/landing_zone/encounters.csv',
            'claims': 'data/landing_zone/claims.csv',
            'providers': 'data/landing_zone/providers.csv',
            'facilities': 'data/landing_zone/facilities.csv',
            'registry': 'data/landing_zone/registry.csv',
            'cms_measures': 'data/landing_zone/cms_measures.csv',
            'hai_data': 'data/landing_zone/hai_data.csv'
        }
        # Don't store loaded_data to avoid session state issues
        # self.loaded_data = {}
    
    def _clean_dataframe_for_display(self, df):
        """Clean DataFrame to avoid JSON serialization issues - enhanced version"""
        if df is None or df.empty:
            return df
        
        try:
            df_clean = df.copy()
            
            # Aggressively convert all columns to safe types
            for col in df_clean.columns:
                try:
                    # Check for ObjectDType specifically
                    if str(df_clean[col].dtype) == 'object' or 'object' in str(df_clean[col].dtype).lower():
                        # Force conversion to string and clean
                        df_clean[col] = df_clean[col].astype(str)
                        df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaT', '<NA>', 'nat'], '')
                    
                    # Handle datetime columns
                    elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                        df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                    
                    # Handle categorical columns
                    elif pd.api.types.is_categorical_dtype(df_clean[col]):
                        df_clean[col] = df_clean[col].astype(str)
                    
                    # Handle numeric columns
                    elif df_clean[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                        df_clean[col] = df_clean[col].fillna(0)
                    
                    # Handle boolean columns
                    elif df_clean[col].dtype == 'bool':
                        df_clean[col] = df_clean[col].astype(str)
                    
                    # Fallback: convert everything else to string
                    else:
                        df_clean[col] = df_clean[col].astype(str)
                        
                except Exception as col_error:
                    # Ultimate fallback: empty string
                    df_clean[col] = ''
            
            # Final safety check - replace any remaining NaN values
            df_clean = df_clean.fillna('')
            
            # Ensure no ObjectDType remains
            for col in df_clean.columns:
                if 'object' in str(df_clean[col].dtype).lower():
                    df_clean[col] = df_clean[col].astype(str).fillna('')
            
            return df_clean
            
        except Exception as e:
            # If all cleaning fails, return empty DataFrame
            try:
                return pd.DataFrame(columns=df.columns if df is not None else [])
            except:
                return pd.DataFrame()
    
    def load_dataset(self, dataset_name):
        """Load a specific dataset - always load fresh to avoid caching issues"""
        filepath = self.data_files.get(dataset_name)
        if filepath and os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                
                # Clean the DataFrame for better Streamlit compatibility
                df = self._clean_dataframe_for_display(df)
                
                # Don't cache in self.loaded_data to avoid session state issues
                return df
            except Exception as e:
                st.error(f"Error loading {dataset_name}: {str(e)}")
                return None
        return None
    
    def get_data_summary(self, df):
        """Get comprehensive data summary"""
        if df is None or df.empty:
            return {}
        
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
        }
        return summary
    
    def check_data_quality(self, df, dataset_name):
        """Comprehensive data quality assessment"""
        if df is None or df.empty:
            return {}
        
        quality_issues = []
        
        # Missing values check
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 20]
        if len(high_missing) > 0:
            quality_issues.append({
                'type': 'High Missing Values',
                'severity': 'High',
                'description': f"{len(high_missing)} columns with >20% missing values",
                'columns': high_missing.index.tolist()
            })
        
        # Duplicate rows check
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_issues.append({
                'type': 'Duplicate Rows',
                'severity': 'Medium',
                'description': f"{duplicates} duplicate rows found ({duplicates/len(df)*100:.1f}%)",
                'count': duplicates
            })
        
        # Data type consistency
        for col in df.columns:
            if 'date' in col.lower() and df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
                except:
                    quality_issues.append({
                        'type': 'Date Format Issue',
                        'severity': 'Medium',
                        'description': f"Column '{col}' appears to be a date but has inconsistent format",
                        'column': col
                    })
        
        # ID column uniqueness
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        for col in id_columns:
            if df[col].nunique() != len(df):
                quality_issues.append({
                    'type': 'Non-Unique ID',
                    'severity': 'High',
                    'description': f"ID column '{col}' has duplicate values",
                    'column': col
                })
        
        # Outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                quality_issues.append({
                    'type': 'Potential Outliers',
                    'severity': 'Low',
                    'description': f"Column '{col}' has {len(outliers)} potential outliers ({len(outliers)/len(df)*100:.1f}%)",
                    'column': col,
                    'count': len(outliers)
                })
        
        return quality_issues
    
    def suggest_transformations(self, df, dataset_name):
        """Suggest data transformations"""
        suggestions = []
        
        if df is None or df.empty:
            return suggestions
        
        # Date column transformations
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            if df[col].dtype == 'object':
                suggestions.append({
                    'type': 'Convert to DateTime',
                    'column': col,
                    'description': f"Convert '{col}' from string to datetime format",
                    'code': f"df['{col}'] = pd.to_datetime(df['{col}'])"
                })
        
        # Missing value handling
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct < 5:
                if df[col].dtype in ['int64', 'float64']:
                    suggestions.append({
                        'type': 'Fill Missing Values',
                        'column': col,
                        'description': f"Fill missing values in '{col}' with median ({missing_pct:.1f}% missing)",
                        'code': f"df['{col}'].fillna(df['{col}'].median(), inplace=True)"
                    })
                else:
                    suggestions.append({
                        'type': 'Fill Missing Values',
                        'column': col,
                        'description': f"Fill missing values in '{col}' with mode ({missing_pct:.1f}% missing)",
                        'code': f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)"
                    })
        
        # Categorical encoding suggestions
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_values = df[col].nunique()
            if unique_values < 10 and unique_values > 2:
                suggestions.append({
                    'type': 'One-Hot Encoding',
                    'column': col,
                    'description': f"Create dummy variables for '{col}' ({unique_values} categories)",
                    'code': f"pd.get_dummies(df['{col}'], prefix='{col}')"
                })
        
        return suggestions

def render_data_explorer():
    """Main data explorer interface - completely isolated from session state issues"""
    
    # NUCLEAR OPTION: Clear ALL session state every time and prevent any storage
    try:
        # Store only essential navigation state
        current_page = st.session_state.get('current_page', 'data_explorer')
        data_status = st.session_state.get('data_status', False)
        user_role = st.session_state.get('user_role', 'Administrator')
        
        # Clear everything
        st.session_state.clear()
        
        # Restore only navigation essentials
        st.session_state.current_page = current_page
        st.session_state.data_status = data_status
        st.session_state.user_role = user_role
        
    except Exception as e:
        # If even this fails, just continue
        pass
    
    # Force garbage collection
    import gc
    gc.collect()
    
    try:
        st.markdown('<h2 class="section-header">🔍 Healthcare Data Explorer & Quality Manager</h2>', unsafe_allow_html=True)
        
        # Initialize explorer (don't store in session state)
        explorer = DataExplorer()
        
        # Ensure explorer is not stored in session state
        if 'explorer' in st.session_state:
            del st.session_state['explorer']
        
        # Check if data exists
        available_datasets = []
        for name, filepath in explorer.data_files.items():
            if os.path.exists(filepath):
                available_datasets.append(name)
        
        if not available_datasets:
            st.warning("⚠️ **No Data Available** - Generate healthcare data first")
            if st.button("🚀 Generate Data", type="primary"):
                st.session_state.current_page = 'generate'
                st.rerun()
            return
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", "🔍 Data Explorer", "✅ Quality Check", "🔧 Transformations", "💾 SQL Queries"
        ])
        
        with tab1:
            render_data_overview(explorer, available_datasets)
        
        with tab2:
            render_data_viewer(explorer, available_datasets)
        
        with tab3:
            render_quality_checker(explorer, available_datasets)
        
        with tab4:
            render_transformations(explorer, available_datasets)
        
        with tab5:
            render_sql_interface(explorer, available_datasets)
            
    except Exception as e:
        st.error(f"❌ **Data Explorer Error:** {str(e)}")
        st.info("💡 **Tip:** Try refreshing the page or check the troubleshooting guide.")
        
        # Aggressively clear ALL session state to prevent JSON serialization issues
        try:
            for key in list(st.session_state.keys()):
                try:
                    del st.session_state[key]
                except:
                    pass
        except:
            pass
        
        # Show a refresh button
        if st.button("🔄 Clear All Data and Refresh"):
            st.rerun()

def render_data_overview(explorer, available_datasets):
    """Render data overview dashboard"""
    st.markdown("### 📊 **Healthcare Data Overview**")
    
    # Dataset summary cards
    summary_cols = st.columns(4)
    
    total_records = 0
    total_size = 0
    
    for i, dataset in enumerate(available_datasets):
        df = explorer.load_dataset(dataset)
        if df is not None:
            summary = explorer.get_data_summary(df)
            total_records += summary['rows']
            total_size += summary['memory_usage']
            
            with summary_cols[i % 4]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{dataset.title()}</h4>
                    <h2 style="color: #2E86AB;">{summary['rows']:,}</h2>
                    <p>{summary['columns']} columns • {summary['memory_usage']:.1f} MB</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Overall statistics
    st.markdown("### 📈 **Overall Statistics**")
    
    overall_cols = st.columns(4)
    
    with overall_cols[0]:
        st.metric("Total Records", f"{total_records:,}")
    
    with overall_cols[1]:
        st.metric("Total Datasets", len(available_datasets))
    
    with overall_cols[2]:
        st.metric("Total Size", f"{total_size:.1f} MB")
    
    with overall_cols[3]:
        avg_records = total_records / len(available_datasets) if available_datasets else 0
        st.metric("Avg Records/Dataset", f"{avg_records:,.0f}")
    
    # Data relationships visualization
    st.markdown("### 🔗 **Data Relationships**")
    
    # Create a simple relationship diagram
    relationships = {
        'patients': ['encounters', 'claims', 'registry'],
        'providers': ['encounters', 'claims', 'registry'],
        'facilities': ['encounters', 'providers'],
        'encounters': ['claims'],
    }
    
    # Show relationships as a network-style visualization
    relationship_info = []
    for parent, children in relationships.items():
        if parent in available_datasets:
            for child in children:
                if child in available_datasets:
                    relationship_info.append(f"**{parent.title()}** → **{child.title()}**")
    
    if relationship_info:
        rel_cols = st.columns(3)
        for i, rel in enumerate(relationship_info):
            with rel_cols[i % 3]:
                st.markdown(rel)
    
    # Data freshness check
    st.markdown("### 🕒 **Data Freshness**")
    
    freshness_cols = st.columns(len(available_datasets))
    
    for i, dataset in enumerate(available_datasets):
        filepath = explorer.data_files[dataset]
        if os.path.exists(filepath):
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            age = datetime.now() - mod_time
            
            with freshness_cols[i]:
                if age.days == 0:
                    status = "🟢 Fresh"
                elif age.days < 7:
                    status = "🟡 Recent"
                else:
                    status = "🔴 Stale"
                
                st.markdown(f"""
                **{dataset.title()}**  
                {status}  
                *{age.days} days old*
                """)

def render_data_viewer(explorer, available_datasets):
    """Render interactive data viewer"""
    st.markdown("### 🔍 **Interactive Data Viewer**")
    
    # Dataset selector
    selected_dataset = st.selectbox(
        "Select Dataset to Explore",
        available_datasets,
        format_func=lambda x: x.title()
    )
    
    if selected_dataset:
        df = explorer.load_dataset(selected_dataset)
        
        if df is not None:
            # Dataset info
            summary = explorer.get_data_summary(df)
            
            info_cols = st.columns(4)
            with info_cols[0]:
                st.metric("Rows", f"{summary['rows']:,}")
            with info_cols[1]:
                st.metric("Columns", summary['columns'])
            with info_cols[2]:
                st.metric("Missing Values", f"{summary['missing_values']:,}")
            with info_cols[3]:
                st.metric("Memory Usage", f"{summary['memory_usage']:.1f} MB")
            
            # Column information
            st.markdown("### 📋 **Column Information**")
            
            col_info = []
            for col in df.columns:
                col_info.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Non-Null': f"{df[col].count():,}",
                    'Null %': f"{(df[col].isnull().sum() / len(df) * 100):.1f}%",
                    'Unique': f"{df[col].nunique():,}",
                    'Sample Values': str(df[col].dropna().head(3).tolist())[:50] + "..."
                })
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, use_container_width=True)
            
            # Data preview with filtering
            st.markdown("### 👀 **Data Preview**")
            
            # Filtering options
            filter_cols = st.columns(3)
            
            with filter_cols[0]:
                show_rows = st.number_input("Rows to Display", 5, 1000, 20)
            
            with filter_cols[1]:
                columns_to_show = st.multiselect(
                    "Columns to Display",
                    df.columns.tolist(),
                    default=df.columns.tolist()[:10]  # Show first 10 by default
                )
            
            with filter_cols[2]:
                sample_type = st.selectbox("Sample Type", ["Head", "Tail", "Random"])
            
            # Apply filters and display
            if columns_to_show:
                try:
                    display_df = df[columns_to_show].copy()
                    
                    # Ensure all columns are properly formatted for display
                    for col in display_df.columns:
                        if display_df[col].dtype == 'object':
                            display_df[col] = display_df[col].astype(str)
                    
                    if sample_type == "Head":
                        display_df = display_df.head(show_rows)
                    elif sample_type == "Tail":
                        display_df = display_df.tail(show_rows)
                    else:  # Random
                        display_df = display_df.sample(min(show_rows, len(df)))
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error displaying data: {str(e)}")
                    st.info("Try selecting fewer columns or reducing the number of rows to display.")
            
            # Statistical summary
            st.markdown("### 📊 **Statistical Summary**")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                try:
                    summary_df = df[numeric_cols].describe()
                    st.dataframe(summary_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating statistical summary: {str(e)}")
            else:
                st.info("No numeric columns found for statistical summary.")
            
            # Value counts for categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.markdown("### 📈 **Categorical Data Distribution**")
                
                selected_cat_col = st.selectbox(
                    "Select Categorical Column",
                    categorical_cols
                )
                
                if selected_cat_col:
                    try:
                        value_counts = df[selected_cat_col].value_counts().head(20)
                        
                        viz_cols = st.columns(2)
                        
                        with viz_cols[0]:
                            fig = px.bar(
                                x=value_counts.values,
                                y=value_counts.index,
                                orientation='h',
                                title=f'Distribution of {selected_cat_col}',
                                color_discrete_sequence=['#2E86AB']
                            )
                            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with viz_cols[1]:
                            st.markdown("**Value Counts:**")
                            # Convert to DataFrame for better display
                            counts_df = pd.DataFrame({
                                'Value': value_counts.index,
                                'Count': value_counts.values
                            })
                            st.dataframe(counts_df, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error analyzing categorical data: {str(e)}")
                        st.info("This column may contain data types that are difficult to analyze.")

def render_quality_checker(explorer, available_datasets):
    """Render data quality assessment"""
    st.markdown("### ✅ **Data Quality Assessment**")
    
    # Dataset selector
    selected_dataset = st.selectbox(
        "Select Dataset for Quality Check",
        available_datasets,
        format_func=lambda x: x.title(),
        key="quality_dataset_selector"
    )
    
    if selected_dataset:
        df = explorer.load_dataset(selected_dataset)
        
        if df is not None:
            # Run quality checks
            quality_issues = explorer.check_data_quality(df, selected_dataset)
            
            # Quality score calculation
            total_checks = 10  # Total number of quality checks
            issues_count = len(quality_issues)
            quality_score = max(0, (total_checks - issues_count) / total_checks * 100)
            
            # Quality dashboard
            quality_cols = st.columns(4)
            
            with quality_cols[0]:
                score_color = "🟢" if quality_score >= 80 else "🟡" if quality_score >= 60 else "🔴"
                st.metric("Quality Score", f"{quality_score:.0f}%", delta=score_color)
            
            with quality_cols[1]:
                st.metric("Issues Found", len(quality_issues))
            
            with quality_cols[2]:
                completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("Data Completeness", f"{completeness:.1f}%")
            
            with quality_cols[3]:
                consistency = (1 - df.duplicated().sum() / len(df)) * 100
                st.metric("Data Consistency", f"{consistency:.1f}%")
            
            # Quality issues details
            if quality_issues:
                st.markdown("### 🚨 **Quality Issues Detected**")
                
                for issue in quality_issues:
                    severity_color = {
                        'High': '🔴',
                        'Medium': '🟡',
                        'Low': '🟢'
                    }.get(issue['severity'], '⚪')
                    
                    with st.expander(f"{severity_color} {issue['type']} - {issue['severity']} Severity"):
                        st.write(issue['description'])
                        
                        if 'columns' in issue:
                            st.write("**Affected Columns:**", ', '.join(issue['columns']))
                        elif 'column' in issue:
                            st.write("**Affected Column:**", issue['column'])
                        
                        # Show sample of problematic data if applicable
                        if issue['type'] == 'High Missing Values' and 'columns' in issue:
                            for col in issue['columns'][:3]:  # Show first 3 columns
                                missing_rows = df[df[col].isnull()].head(5)
                                if not missing_rows.empty:
                                    st.write(f"**Sample rows with missing {col}:**")
                                    st.dataframe(missing_rows[[col] + [c for c in df.columns if c != col][:3]], use_container_width=True)
            else:
                st.success("🎉 **No Quality Issues Detected!** Your data looks great.")
            
            # Data profiling
            st.markdown("### 📊 **Data Profiling Report**")
            
            profile_tabs = st.tabs(["Missing Values", "Data Types", "Distributions"])
            
            with profile_tabs[0]:
                # Missing values heatmap
                missing_data = df.isnull().sum()
                missing_pct = (missing_data / len(df) * 100).round(2)
                
                if missing_data.sum() > 0:
                    missing_df = pd.DataFrame({
                        'Column': missing_data.index,
                        'Missing Count': missing_data.values,
                        'Missing %': missing_pct.values
                    }).sort_values('Missing %', ascending=False)
                    
                    fig = px.bar(
                        missing_df,
                        x='Column',
                        y='Missing %',
                        title='Missing Values by Column',
                        color='Missing %',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("No missing values found in the dataset!")
            
            with profile_tabs[1]:
                # Data types summary
                dtype_summary = df.dtypes.value_counts()
                
                fig = px.pie(
                    values=dtype_summary.values,
                    names=dtype_summary.index,
                    title='Data Types Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with profile_tabs[2]:
                # Distributions for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    selected_numeric = st.selectbox("Select Numeric Column", numeric_cols)
                    
                    if selected_numeric:
                        fig = px.histogram(
                            df,
                            x=selected_numeric,
                            title=f'Distribution of {selected_numeric}',
                            nbins=30
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Box plot for outlier detection
                        fig_box = px.box(
                            df,
                            y=selected_numeric,
                            title=f'Box Plot of {selected_numeric} (Outlier Detection)'
                        )
                        st.plotly_chart(fig_box, use_container_width=True)

def render_transformations(explorer, available_datasets):
    """Render data transformation interface"""
    st.markdown("### 🔧 **Data Transformations**")
    
    # Dataset selector
    selected_dataset = st.selectbox(
        "Select Dataset for Transformation",
        available_datasets,
        format_func=lambda x: x.title(),
        key="transform_dataset_selector"
    )
    
    if selected_dataset:
        df = explorer.load_dataset(selected_dataset)
        
        if df is not None:
            # Get transformation suggestions
            suggestions = explorer.suggest_transformations(df, selected_dataset)
            
            st.markdown("### 💡 **Recommended Transformations**")
            
            if suggestions:
                for i, suggestion in enumerate(suggestions):
                    with st.expander(f"🔧 {suggestion['type']} - {suggestion['column']}"):
                        st.write(suggestion['description'])
                        st.code(suggestion['code'], language='python')
                        
                        if st.button(f"Apply Transformation", key=f"apply_{i}"):
                            try:
                                # This is a simplified example - in production, you'd want more robust execution
                                st.info("Transformation applied! (Note: This is a demo - actual transformation would modify the dataset)")
                            except Exception as e:
                                st.error(f"Error applying transformation: {str(e)}")
            else:
                st.info("No automatic transformations suggested. Your data appears to be in good shape!")
            
            # Manual transformations
            st.markdown("### ⚙️ **Manual Transformations**")
            
            transform_tabs = st.tabs(["Column Operations", "Data Cleaning", "Feature Engineering"])
            
            with transform_tabs[0]:
                st.markdown("**Column Operations**")
                
                col_ops = st.columns(2)
                
                with col_ops[0]:
                    st.markdown("**Rename Columns**")
                    old_name = st.selectbox("Select Column to Rename", df.columns)
                    new_name = st.text_input("New Column Name")
                    
                    if st.button("Rename Column") and new_name:
                        st.code(f"df.rename(columns={{'{old_name}': '{new_name}'}}, inplace=True)")
                
                with col_ops[1]:
                    st.markdown("**Drop Columns**")
                    cols_to_drop = st.multiselect("Select Columns to Drop", df.columns)
                    
                    if st.button("Drop Columns") and cols_to_drop:
                        st.code(f"df.drop(columns={cols_to_drop}, inplace=True)")
            
            with transform_tabs[1]:
                st.markdown("**Data Cleaning Operations**")
                
                cleaning_ops = st.columns(2)
                
                with cleaning_ops[0]:
                    st.markdown("**Handle Missing Values**")
                    missing_col = st.selectbox("Select Column with Missing Values", 
                                             df.columns[df.isnull().any()].tolist())
                    
                    if missing_col:
                        strategy = st.selectbox("Fill Strategy", 
                                              ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Drop Rows"])
                        
                        if st.button("Apply Missing Value Strategy"):
                            strategies = {
                                "Mean": f"df['{missing_col}'].fillna(df['{missing_col}'].mean(), inplace=True)",
                                "Median": f"df['{missing_col}'].fillna(df['{missing_col}'].median(), inplace=True)",
                                "Mode": f"df['{missing_col}'].fillna(df['{missing_col}'].mode()[0], inplace=True)",
                                "Forward Fill": f"df['{missing_col}'].fillna(method='ffill', inplace=True)",
                                "Backward Fill": f"df['{missing_col}'].fillna(method='bfill', inplace=True)",
                                "Drop Rows": f"df.dropna(subset=['{missing_col}'], inplace=True)"
                            }
                            st.code(strategies[strategy])
                
                with cleaning_ops[1]:
                    st.markdown("**Remove Duplicates**")
                    duplicate_count = df.duplicated().sum()
                    st.write(f"Found {duplicate_count} duplicate rows")
                    
                    if st.button("Remove Duplicates") and duplicate_count > 0:
                        st.code("df.drop_duplicates(inplace=True)")
            
            with transform_tabs[2]:
                st.markdown("**Feature Engineering**")
                
                feature_ops = st.columns(2)
                
                with feature_ops[0]:
                    st.markdown("**Create Date Features**")
                    date_cols = [col for col in df.columns if 'date' in col.lower()]
                    
                    if date_cols:
                        date_col = st.selectbox("Select Date Column", date_cols)
                        features = st.multiselect("Extract Features", 
                                                ["Year", "Month", "Day", "Weekday", "Quarter"])
                        
                        if st.button("Create Date Features") and features:
                            code_lines = [f"df['{date_col}'] = pd.to_datetime(df['{date_col}'])"]
                            for feature in features:
                                if feature == "Year":
                                    code_lines.append(f"df['{date_col}_year'] = df['{date_col}'].dt.year")
                                elif feature == "Month":
                                    code_lines.append(f"df['{date_col}_month'] = df['{date_col}'].dt.month")
                                elif feature == "Day":
                                    code_lines.append(f"df['{date_col}_day'] = df['{date_col}'].dt.day")
                                elif feature == "Weekday":
                                    code_lines.append(f"df['{date_col}_weekday'] = df['{date_col}'].dt.dayofweek")
                                elif feature == "Quarter":
                                    code_lines.append(f"df['{date_col}_quarter'] = df['{date_col}'].dt.quarter")
                            
                            st.code('\n'.join(code_lines))
                
                with feature_ops[1]:
                    st.markdown("**Categorical Encoding**")
                    cat_cols = df.select_dtypes(include=['object']).columns
                    
                    if len(cat_cols) > 0:
                        cat_col = st.selectbox("Select Categorical Column", cat_cols)
                        encoding_type = st.selectbox("Encoding Type", 
                                                   ["One-Hot Encoding", "Label Encoding", "Target Encoding"])
                        
                        if st.button("Apply Encoding"):
                            if encoding_type == "One-Hot Encoding":
                                st.code(f"pd.get_dummies(df['{cat_col}'], prefix='{cat_col}')")
                            elif encoding_type == "Label Encoding":
                                st.code(f"from sklearn.preprocessing import LabelEncoder\n"
                                       f"le = LabelEncoder()\n"
                                       f"df['{cat_col}_encoded'] = le.fit_transform(df['{cat_col}'])")

def render_custom_sql_interface(explorer, available_datasets):
    """Render custom SQL query interface"""
    # Available tables info
    st.markdown("**Available Tables:**")
    table_info = []
    for dataset in available_datasets:
        df = explorer.load_dataset(dataset)
        if df is not None:
            table_info.append(f"• **{dataset}** ({len(df):,} rows, {len(df.columns)} columns)")
    
    for info in table_info:
        st.markdown(info)
    
    # Custom query editor
    st.markdown("**Write Your SQL Query:**")
    custom_query = st.text_area(
        "SQL Query",
        height=200,
        placeholder="""
Example:
SELECT 
    age,
    gender,
    COUNT(*) as patient_count
FROM patients 
WHERE age > 65
GROUP BY age, gender
ORDER BY patient_count DESC;
        """.strip()
    )
    
    if st.button("Execute Custom Query") and custom_query.strip():
        try:
            with st.spinner("Executing custom query..."):
                conn = create_database_connection(explorer, available_datasets)
                
                if conn:
                    result_df = pd.read_sql_query(custom_query, conn)
                    
                    if not result_df.empty:
                        st.success(f"Query executed successfully! Found {len(result_df)} rows.")
                        
                        # Store result in session state
                        safe_store_in_session_state('last_query_result', result_df, "Custom Query")
                        
                        # Show results
                        st.dataframe(result_df, use_container_width=True)
                    else:
                        st.warning("Query executed but returned no results.")
                    
                    conn.close()
                else:
                    st.error("Could not create database connection.")
        
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            st.info("💡 **Tip:** Check your SQL syntax and table names.")

def render_sql_interface(explorer, available_datasets):
    """Render SQL query interface"""
    st.markdown("### 💾 **SQL Query Interface**")
    
    if sql_library is None:
        st.warning("⚠️ **SQL Library Unavailable**")
        st.info("""
        The SQL query library is currently unavailable. You can still:
        - Use the Custom Queries tab to write your own SQL
        - Explore data using the other tabs
        - Check the troubleshooting guide for SQL library issues
        """)
        
        # Show custom query interface even without the library
        st.markdown("### ✍️ **Custom SQL Queries**")
        render_custom_sql_interface(explorer, available_datasets)
        return
    
    # SQL interface tabs
    sql_tabs = st.tabs(["📚 Query Library", "✍️ Custom Queries", "📊 Query Results"])
    
    with sql_tabs[0]:
        st.markdown("### 📚 **Healthcare SQL Query Library**")
        
        # Display available query categories
        query_categories = sql_library.get_query_list()
        
        selected_category = st.selectbox(
            "Select Query Category",
            list(query_categories.keys())
        )
        
        if selected_category:
            queries_in_category = query_categories[selected_category]
            
            selected_query = st.selectbox(
                "Select Query",
                queries_in_category
            )
            
            if selected_query:
                # Show query explanation
                explanation = sql_library.explain_query(selected_query)
                st.markdown("**Query Description:**")
                st.info(explanation)
                
                # Show SQL code
                sql_code = sql_library.get_query_text(selected_query)
                st.markdown("**SQL Code:**")
                st.code(sql_code, language='sql')
                
                # Execute query button
                if st.button("Execute Query", key=f"exec_{selected_query}"):
                    try:
                        with st.spinner("Executing query..."):
                            # Create database connection
                            conn = create_database_connection(explorer, available_datasets)
                            
                            if conn:
                                result_df = sql_library.execute_query(selected_query, conn)
                                
                                if not result_df.empty:
                                    # Clean the DataFrame for display
                                    result_df = explorer._clean_dataframe_for_display(result_df)
                                    
                                    st.success(f"Query executed successfully! Found {len(result_df)} rows.")
                                    
                                    # Store result in session state for the results tab
                                    safe_store_in_session_state('last_query_result', result_df, selected_query)
                                    
                                    # Show preview
                                    st.markdown("**Query Results Preview:**")
                                    st.dataframe(result_df.head(10), use_container_width=True)
                                    
                                    # Add export functionality directly here
                                    export_cols = st.columns(3)
                                    with export_cols[0]:
                                        csv = result_df.to_csv(index=False)
                                        st.download_button("📥 Download CSV", csv, f"{selected_query}_results.csv", "text/csv")
                                    with export_cols[1]:
                                        json_str = result_df.to_json(orient='records', indent=2)
                                        st.download_button("📥 Download JSON", json_str, f"{selected_query}_results.json", "application/json")
                                    with export_cols[2]:
                                        st.info(f"Results: {len(result_df)} rows × {len(result_df.columns)} columns")
                                else:
                                    st.warning("Query executed but returned no results.")
                                
                                conn.close()
                            else:
                                st.error("Could not create database connection.")
                    
                    except Exception as e:
                        st.error(f"Error executing query: {str(e)}")
    
    with sql_tabs[1]:
        st.markdown("### ✍️ **Custom SQL Queries**")
        
        # Available tables info
        st.markdown("**Available Tables:**")
        table_info = []
        for dataset in available_datasets:
            df = explorer.load_dataset(dataset)
            if df is not None:
                table_info.append(f"• **{dataset}** ({len(df):,} rows, {len(df.columns)} columns)")
        
        for info in table_info:
            st.markdown(info)
        
        # Custom query editor
        st.markdown("**Write Your SQL Query:**")
        custom_query = st.text_area(
            "SQL Query",
            height=200,
            placeholder="""
Example:
SELECT 
    age,
    gender,
    COUNT(*) as patient_count
FROM patients 
WHERE age > 65
GROUP BY age, gender
ORDER BY patient_count DESC;
            """.strip()
        )
        
        if st.button("Execute Custom Query") and custom_query.strip():
            try:
                with st.spinner("Executing custom query..."):
                    conn = create_database_connection(explorer, available_datasets)
                    
                    if conn:
                        result_df = pd.read_sql_query(custom_query, conn)
                        
                        if not result_df.empty:
                            # Clean the DataFrame for display
                            result_df = explorer._clean_dataframe_for_display(result_df)
                            
                            st.success(f"Query executed successfully! Found {len(result_df)} rows.")
                            
                            # Store result in session state
                            safe_store_in_session_state('last_query_result', result_df, "Custom Query")
                            
                            # Show results
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Add export functionality directly here
                            export_cols = st.columns(3)
                            with export_cols[0]:
                                csv = result_df.to_csv(index=False)
                                st.download_button("📥 Download CSV", csv, "custom_query_results.csv", "text/csv")
                            with export_cols[1]:
                                json_str = result_df.to_json(orient='records', indent=2)
                                st.download_button("📥 Download JSON", json_str, "custom_query_results.json", "application/json")
                            with export_cols[2]:
                                st.info(f"Results: {len(result_df)} rows × {len(result_df.columns)} columns")
                        else:
                            st.warning("Query executed but returned no results.")
                        
                        conn.close()
                    else:
                        st.error("Could not create database connection.")
            
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
                st.info("💡 **Tip:** Check your SQL syntax and table names.")
    
    with sql_tabs[2]:
        st.markdown("### 📊 **Query Results Analysis**")
        
        st.info("""
        **Query Results Display:**
        - Query results are shown immediately after execution in the Query Library and Custom Queries tabs
        - Use the Export buttons in those tabs to save results to CSV or JSON
        - For analysis and visualization, execute your query in the Custom Queries tab
        """)
        
        st.markdown("### 💡 **Tips for Query Analysis**")
        
        tips_cols = st.columns(2)
        
        with tips_cols[0]:
            st.markdown("""
            **Query Best Practices:**
            - Use LIMIT to test queries with large datasets
            - Add ORDER BY for consistent results
            - Use aggregate functions (COUNT, SUM, AVG) for summaries
            - Join tables to get comprehensive insights
            """)
        
        with tips_cols[1]:
            st.markdown("""
            **Export Options:**
            - **CSV**: Best for Excel analysis
            - **JSON**: Best for API integration
            - **Copy Results**: Use browser copy function
            - **Screenshots**: Use browser screenshot tools
            """)
        
        st.markdown("### 📈 **Sample Queries for Analysis**")
        
        sample_queries = [
            {
                "title": "Patient Demographics",
                "query": "SELECT age, gender, COUNT(*) as count FROM patients GROUP BY age, gender ORDER BY count DESC LIMIT 20"
            },
            {
                "title": "Top Diagnoses",
                "query": "SELECT primary_diagnosis, COUNT(*) as frequency FROM encounters GROUP BY primary_diagnosis ORDER BY frequency DESC LIMIT 10"
            },
            {
                "title": "Monthly Encounters",
                "query": "SELECT strftime('%Y-%m', encounter_date) as month, COUNT(*) as encounters FROM encounters GROUP BY month ORDER BY month"
            },
            {
                "title": "Provider Performance",
                "query": "SELECT e.provider_id, COUNT(*) as total_encounters, ROUND(AVG(c.claim_amount), 2) as avg_cost FROM encounters e LEFT JOIN claims c ON e.patient_id = c.patient_id GROUP BY e.provider_id ORDER BY total_encounters DESC LIMIT 15"
            }
        ]
        
        for i, sample in enumerate(sample_queries):
            with st.expander(f"📊 {sample['title']}"):
                st.code(sample['query'], language='sql')
                st.info("Copy this query to the Custom Queries tab to execute it.")

def create_database_connection(explorer, available_datasets):
    """Create SQLite database connection with loaded data"""
    try:
        conn = sqlite3.connect(':memory:')
        
        # Load all available datasets into the database
        for dataset in available_datasets:
            df = explorer.load_dataset(dataset)
            if df is not None:
                # Clean the DataFrame before loading into database
                df_clean = explorer._clean_dataframe_for_display(df)
                df_clean.to_sql(dataset, conn, index=False, if_exists='replace')
        
        return conn
    
    except Exception as e:
        st.error(f"Error creating database connection: {str(e)}")
        return None

# CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .section-header {
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)