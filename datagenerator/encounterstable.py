def generate_encounters(patients):
    rows = []
    for _, p in patients.iterrows():
        for _ in range(random.randint(1, AVG_ENCOUNTERS_PER_PATIENT)):
            admit = random_date(START_DATE, END_DATE)
            los = random.randint(1, 10)
            rows.append({
                "encounter_id": str(uuid.uuid4()),
                "patient_id": p.patient_id,
                "hospital": random.choice(HOSPITALS),
                "admit_date": admit,
                "discharge_date": admit + timedelta(days=los),
                "length_of_stay": los
            })
    return pd.DataFrame(rows)
