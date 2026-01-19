def generate_hai(encounters):
    rows = []
    for h in HOSPITALS:
        rows.append({
            "hospital": h,
            "reporting_month": random_date(START_DATE, END_DATE),
            "hai_rate": round(random.uniform(0.5, 5.0), 2)
        })
    return pd.DataFrame(rows)
