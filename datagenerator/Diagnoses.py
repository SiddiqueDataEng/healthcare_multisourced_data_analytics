def generate_diagnoses(encounters):
    rows = []
    for _, e in encounters.iterrows():
        for _ in range(random.randint(1,3)):
            code = random.choice(list(DIAGNOSES.keys()))
            rows.append({
                "encounter_id": e.encounter_id,
                "icd_code": code,
                "description": DIAGNOSES[code]
            })
    return pd.DataFrame(rows)
