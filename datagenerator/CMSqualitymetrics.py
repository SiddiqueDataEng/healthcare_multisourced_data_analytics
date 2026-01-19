def generate_cms_metrics():
    return pd.DataFrame({
        "hospital": HOSPITALS,
        "readmission_rate": np.random.uniform(10, 25, len(HOSPITALS)),
        "mortality_rate": np.random.uniform(1, 5, len(HOSPITALS)),
        "patient_satisfaction": np.random.uniform(60, 95, len(HOSPITALS))
    })
