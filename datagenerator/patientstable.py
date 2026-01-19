def generate_patients(n):
    return pd.DataFrame({
        "patient_id": [str(uuid.uuid4()) for _ in range(n)],
        "gender": np.random.choice(["M", "F"], n),
        "age": np.random.randint(0, 95, n),
        "race": np.random.choice(["White", "Black", "Asian", "Hispanic", "Other"], n),
        "date_of_birth": [
            datetime.now() - timedelta(days=365*age) for age in np.random.randint(0,95,n)
        ]
    })
