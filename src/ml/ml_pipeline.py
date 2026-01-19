"""
Comprehensive Machine Learning Pipeline for Healthcare Analytics
Includes readmission prediction, cost prediction, and feature engineering
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report
)
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class MLPipeline:
    """Comprehensive ML pipeline for healthcare predictions"""
    
    def __init__(self, model_dir: str = "models"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def engineer_readmission_features(self, patients_df: pd.DataFrame, 
                                     encounters_df: pd.DataFrame,
                                     claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for readmission prediction
        
        Features:
            - Patient demographics (age, gender, race)
            - Comorbidity count and specific conditions
            - Prior utilization (encounter count, claim count)
            - Length of stay
            - Severity level
            - Cost metrics
            - Time since last encounter
        """
        self.logger.info("Engineering features for readmission prediction")
        
        # Start with encounters
        features_df = encounters_df.copy()
        
        # Merge patient data
        features_df = features_df.merge(
            patients_df[['patient_id', 'age', 'gender', 'race', 'comorbidity_count', 
                        'risk_score', 'is_high_cost']],
            on='patient_id',
            how='left'
        )
        
        # Calculate prior utilization
        encounter_counts = encounters_df.groupby('patient_id').size().reset_index(name='prior_encounter_count')
        features_df = features_df.merge(encounter_counts, on='patient_id', how='left')
        
        claim_counts = claims_df.groupby('patient_id').size().reset_index(name='prior_claim_count')
        features_df = features_df.merge(claim_counts, on='patient_id', how='left')
        
        # Calculate total costs
        patient_costs = claims_df.groupby('patient_id')['claim_amount'].sum().reset_index(name='total_claim_cost')
        features_df = features_df.merge(patient_costs, on='patient_id', how='left')
        
        # Encode categorical variables
        features_df['gender_encoded'] = self._encode_column(features_df, 'gender')
        features_df['race_encoded'] = self._encode_column(features_df, 'race')
        features_df['encounter_type_encoded'] = self._encode_column(features_df, 'encounter_type')
        features_df['severity_encoded'] = self._encode_column(features_df, 'severity_level')
        
        # Binary flags
        features_df['is_inpatient'] = (features_df['encounter_type'] == 'Inpatient').astype(int)
        features_df['is_emergency'] = (features_df['encounter_type'] == 'Emergency').astype(int)
        features_df['is_severe'] = (features_df['severity_level'].isin(['Severe', 'Critical'])).astype(int)
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        self.logger.info(f"Engineered {len(features_df)} samples with {len(features_df.columns)} features")
        return features_df
    
    def train_readmission_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train multiple models for readmission prediction and compare performance
        
        Models:
            - Logistic Regression
            - Random Forest
            - Gradient Boosting
        
        Returns:
            - Model performance metrics
            - Feature importance
            - Best model
        """
        self.logger.info("Training readmission prediction models")
        
        # Select features
        feature_cols = [
            'age', 'gender_encoded', 'race_encoded', 'comorbidity_count', 'risk_score',
            'prior_encounter_count', 'prior_claim_count', 'total_claim_cost',
            'length_of_stay', 'encounter_type_encoded', 'severity_encoded',
            'is_inpatient', 'is_emergency', 'is_severe', 'is_high_cost'
        ]
        
        # Filter to available features
        feature_cols = [col for col in feature_cols if col in features_df.columns]
        
        X = features_df[feature_cols]
        y = features_df['is_readmission'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
        }
        
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            
            # Train
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            results[name] = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                importance = np.zeros(len(feature_cols))
            
            feature_importance = sorted(
                zip(feature_cols, importance),
                key=lambda x: x[1],
                reverse=True
            )
            results[name]['feature_importance'] = [
                {'feature': feat, 'importance': float(imp)} 
                for feat, imp in feature_importance[:10]
            ]
            
            # Save best model
            if name == 'Gradient Boosting':  # Usually best for healthcare
                self._save_model(model, self.model_dir / 'readmission_model.pkl')
                results[name]['saved'] = True
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
        
        summary = {
            'models': results,
            'best_model': best_model[0],
            'best_roc_auc': best_model[1]['roc_auc'],
            'feature_columns': feature_cols,
            'train_size': int(len(X_train)),
            'test_size': int(len(X_test)),
            'positive_class_rate': float(y.mean())
        }
        
        self.logger.info(f"Best model: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.3f})")
        return summary
    
    def engineer_cost_features(self, patients_df: pd.DataFrame,
                               encounters_df: pd.DataFrame,
                               claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for cost prediction
        
        Features:
            - Patient demographics and risk
            - Encounter characteristics
            - Diagnosis and procedure codes
            - Historical utilization
            - Comorbidities
        """
        self.logger.info("Engineering features for cost prediction")
        
        # Aggregate claims by encounter
        encounter_costs = claims_df.groupby('patient_id').agg({
            'claim_amount': 'sum',
            'paid_amount': 'sum'
        }).reset_index()
        encounter_costs.columns = ['patient_id', 'total_claim_amount', 'total_paid_amount']
        
        # Start with patients
        features_df = patients_df.copy()
        
        # Merge costs
        features_df = features_df.merge(encounter_costs, on='patient_id', how='left')
        
        # Calculate encounter metrics
        encounter_metrics = encounters_df.groupby('patient_id').agg({
            'encounter_id': 'count',
            'length_of_stay': 'mean',
            'is_readmission': 'sum'
        }).reset_index()
        encounter_metrics.columns = ['patient_id', 'encounter_count', 'avg_los', 'readmission_count']
        
        features_df = features_df.merge(encounter_metrics, on='patient_id', how='left')
        
        # Encode categorical
        features_df['gender_encoded'] = self._encode_column(features_df, 'gender')
        features_df['race_encoded'] = self._encode_column(features_df, 'race')
        
        # Fill missing
        features_df = features_df.fillna(0)
        
        # Target variable
        features_df['cost_per_patient'] = features_df['total_claim_amount']
        
        self.logger.info(f"Engineered cost features for {len(features_df)} patients")
        return features_df
    
    def train_cost_prediction_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train regression models to predict healthcare costs
        
        Models:
            - Linear Regression (Ridge)
            - Random Forest Regressor
            - Gradient Boosting Regressor
        
        Returns:
            - Model performance metrics (RMSE, MAE, R²)
            - Feature importance
            - Cost distribution analysis
        """
        self.logger.info("Training cost prediction models")
        
        # Select features
        feature_cols = [
            'age', 'gender_encoded', 'race_encoded', 'comorbidity_count', 'risk_score',
            'encounter_count', 'avg_los', 'readmission_count', 'is_high_cost'
        ]
        
        feature_cols = [col for col in feature_cols if col in features_df.columns]
        
        # Filter out patients with no costs
        features_df = features_df[features_df['cost_per_patient'] > 0].copy()
        
        X = features_df[feature_cols]
        y = features_df['cost_per_patient']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models = {
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        }
        
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            
            # Train
            if name == 'Ridge Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2_score': float(r2),
                'mean_actual': float(y_test.mean()),
                'mean_predicted': float(y_pred.mean()),
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                importance = np.zeros(len(feature_cols))
            
            feature_importance = sorted(
                zip(feature_cols, importance),
                key=lambda x: x[1],
                reverse=True
            )
            results[name]['feature_importance'] = [
                {'feature': feat, 'importance': float(imp)}
                for feat, imp in feature_importance
            ]
            
            # Save best model
            if name == 'Gradient Boosting':
                self._save_model(model, self.model_dir / 'cost_prediction_model.pkl')
                results[name]['saved'] = True
        
        # Find best model (by R²)
        best_model = max(results.items(), key=lambda x: x[1]['r2_score'])
        
        # Cost distribution analysis
        cost_distribution = {
            'min': float(y.min()),
            'max': float(y.max()),
            'mean': float(y.mean()),
            'median': float(y.median()),
            'std': float(y.std()),
            'q25': float(y.quantile(0.25)),
            'q75': float(y.quantile(0.75)),
        }
        
        summary = {
            'models': results,
            'best_model': best_model[0],
            'best_r2_score': best_model[1]['r2_score'],
            'cost_distribution': cost_distribution,
            'feature_columns': feature_cols,
            'train_size': int(len(X_train)),
            'test_size': int(len(X_test)),
        }
        
        self.logger.info(f"Best model: {best_model[0]} (R²: {best_model[1]['r2_score']:.3f})")
        return summary
    
    def _encode_column(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Encode categorical column"""
        if col not in self.label_encoders:
            self.label_encoders[col] = LabelEncoder()
            return pd.Series(self.label_encoders[col].fit_transform(df[col].astype(str)))
        else:
            return pd.Series(self.label_encoders[col].transform(df[col].astype(str)))
    
    def _save_model(self, model: Any, path: Path) -> None:
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> Any:
        """Load model from disk"""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self.logger.info(f"Model loaded from {path}")
        return model
