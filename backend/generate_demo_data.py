"""Generate clearly labelled synthetic daily sales history for NetworkIQ demos.
This is for development/testing only; it is not real retailer data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_INVENTORY = ROOT.parent / 'inventory_data_large.csv'
DEFAULT_OUTPUT = ROOT / 'data' / 'synthetic_sales_history.csv'

def generate(inventory_path: Path = DEFAULT_INVENTORY, output_path: Path = DEFAULT_OUTPUT, days: int = 365, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inv = pd.read_csv(inventory_path)
    required = {'Store','Product','Category','Demand','Price'}
    missing = required - set(inv.columns)
    if missing:
        raise ValueError(f'Inventory CSV is missing columns: {sorted(missing)}')
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=days, freq='D')
    rows = []
    for _, item in inv.iterrows():
        base = max(1.0, float(item['Demand']) / 7.0)
        store = str(item['Store']); product = str(item['Product']); category = str(item['Category'])
        product_bias = rng.uniform(0.78, 1.24)
        for date in dates:
            weekday = 1.18 if date.dayofweek in (4, 5) else (0.86 if date.dayofweek == 0 else 1.0)
            annual = 1 + 0.12 * np.sin(2 * np.pi * date.dayofyear / 365.25)
            month = 1.12 if date.month in (10, 11, 12) else 1.0
            promo = int(rng.random() < 0.12)
            promo_factor = 1.30 if promo else 1.0
            price = float(item['Price']) * (0.95 if promo else 1.0)
            noise = rng.lognormal(mean=0, sigma=0.18)
            units = max(0, int(round(base * product_bias * weekday * annual * month * promo_factor * noise)))
            rows.append({
                'Date': date.strftime('%Y-%m-%d'), 'Store': store, 'Product': product,
                'Category': category, 'UnitsSold': units, 'Price': round(price, 2),
                'Promotion': promo, 'Source': 'synthetic_demo'
            })
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    df = generate(args.inventory, args.output, args.days, args.seed)
    print(f'Generated {len(df):,} synthetic daily sales records at {args.output}')
