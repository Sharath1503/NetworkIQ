from __future__ import annotations
import os, math, json, re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import linprog
import joblib
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / '.env')
DATA_DIR = HERE / 'data'; MODEL_DIR = HERE / 'models'
DATA_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
HISTORY_PATH = DATA_DIR / 'sales_history.csv'
SYNTHETIC_PATH = DATA_DIR / 'synthetic_sales_history.csv'
INVENTORY_PATH = HERE.parent / 'inventory_data_large.csv'
FEATURES = ['Store','Product','Category','Price','Promotion','day_of_week','month','lag_1','lag_7','lag_14','rolling_7','rolling_14']
CAT_FEATURES = ['Store','Product','Category']
NUM_FEATURES = [c for c in FEATURES if c not in CAT_FEATURES]

app = FastAPI(title='NetworkIQ Intelligence API', version='1.0.0', description='Demand forecasting, stockout risk, transfer optimization, and inventory assistant.')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173','http://localhost:4173'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

class InventoryItem(BaseModel):
    Store: str; Product: str; Category: str = 'Other'; CurrentStock: float = Field(ge=0)
    Demand: float = Field(default=1, ge=0); Price: float = Field(default=0, ge=0)
    HoldingCost: float = Field(default=0, ge=0); TransferCost: float = Field(default=0, ge=0); Location: str = ''

class InventoryRequest(BaseModel):
    inventory: list[InventoryItem] = Field(min_length=1)
    horizon_days: int = Field(default=7, ge=1, le=30)

class AssistantRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    inventory: list[InventoryItem] = Field(default_factory=list)
    forecasts: list[dict[str, Any]] = Field(default_factory=list)
    transfers: list[dict[str, Any]] = Field(default_factory=list)

state: dict[str, Any] = {'regressor': None, 'classifier': None, 'metrics': None, 'feature_importance': [], 'history_source': None, 'history_rows': 0}

def _inventory_df(items: list[InventoryItem] | list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([x.model_dump() if isinstance(x, InventoryItem) else x for x in items])
    if df.empty: raise HTTPException(400, 'Inventory is empty.')
    for c in ['CurrentStock','Demand','Price','HoldingCost','TransferCost']:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0).clip(lower=0)
    for c in ['Store','Product','Category']:
        df[c] = df.get(c, 'Unknown').fillna('Unknown').astype(str)
    return df

def _build_features(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date','Store','Product','UnitsSold']).sort_values(['Store','Product','Date'])
    df['UnitsSold'] = pd.to_numeric(df['UnitsSold'], errors='coerce').fillna(0).clip(lower=0)
    df['Price'] = pd.to_numeric(df.get('Price', 0), errors='coerce').fillna(0)
    df['Promotion'] = pd.to_numeric(df.get('Promotion', 0), errors='coerce').fillna(0)
    grouped = df.groupby(['Store','Product'], sort=False)['UnitsSold']
    df['lag_1'] = grouped.shift(1)
    df['lag_7'] = grouped.shift(7)
    df['lag_14'] = grouped.shift(14)
    df['rolling_7'] = df.groupby(['Store','Product'], sort=False)['UnitsSold'].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    df['rolling_14'] = df.groupby(['Store','Product'], sort=False)['UnitsSold'].transform(lambda s: s.shift(1).rolling(14, min_periods=5).mean())
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['Category'] = df.get('Category', 'Other').fillna('Other').astype(str)
    df['Store'] = df['Store'].astype(str); df['Product'] = df['Product'].astype(str)
    return df.dropna(subset=['lag_1','lag_7','lag_14','rolling_7','rolling_14']).reset_index(drop=True)

def _pipeline(model):
    prep = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES), ('num','passthrough',NUM_FEATURES)])
    return Pipeline([('preprocess', prep), ('model', model)])

def _ensure_history(inventory: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    # Prefer user-provided history, otherwise a clearly-labelled synthetic demo history.
    if HISTORY_PATH.exists():
        df = pd.read_csv(HISTORY_PATH)
        return df, 'uploaded_or_user_provided'
    if SYNTHETIC_PATH.exists():
        return pd.read_csv(SYNTHETIC_PATH), 'synthetic_demo'
    from generate_demo_data import generate
    temp_inventory = HERE.parent / 'inventory_data_large.csv'
    if not temp_inventory.exists():
        inventory[['Store','Product','Category','CurrentStock','Demand','Price','HoldingCost','TransferCost','Location']].to_csv(temp_inventory, index=False)
    df = generate(temp_inventory, SYNTHETIC_PATH, 365, 42)
    return df, 'synthetic_demo'

def _train(history: pd.DataFrame, source: str) -> dict:
    required = {'Date','Store','Product','UnitsSold'}
    if not required.issubset(history.columns):
        raise HTTPException(400, 'Sales history must include Date, Store, Product, UnitsSold. Optional: Category, Price, Promotion.')
    history = history.copy()
    if 'Category' not in history: history['Category'] = 'Other'
    if 'Price' not in history: history['Price'] = 0.0
    if 'Promotion' not in history: history['Promotion'] = 0
    engineered = _build_features(history)
    if len(engineered) < 50: raise HTTPException(400, 'At least 50 usable rows are needed; 3+ weeks per product/store is recommended.')
    # Chronological holdout: last 20% of dates are kept out of model fitting.
    dates = sorted(engineered['Date'].unique())
    split_idx = max(1, int(len(dates) * 0.8))
    split_date = dates[split_idx] if split_idx < len(dates) else dates[-1]
    train = engineered[engineered['Date'] < split_date]
    test = engineered[engineered['Date'] >= split_date]
    if len(train) < 20 or len(test) < 10:
        cut = int(len(engineered) * 0.8); train, test = engineered.iloc[:cut], engineered.iloc[cut:]
    model = _pipeline(RandomForestRegressor(n_estimators=70, min_samples_leaf=2, max_features=0.9, random_state=42, n_jobs=2))
    model.fit(train[FEATURES], train['UnitsSold'])
    pred = np.maximum(0, model.predict(test[FEATURES]))
    actual = test['UnitsSold'].to_numpy()
    mae = float(mean_absolute_error(actual, pred)); rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    denom = float(np.sum(np.abs(actual)))
    wape = float(np.sum(np.abs(actual - pred)) / denom * 100) if denom else 0.0
    # Synthetic/derived risk labels are used only as a demo classifier. For real use, supply historical stockout labels.
    risk_target = (train['lag_1'] + train['lag_7'] + train['lag_14'] + train['rolling_7'] * 4 >= np.maximum(1, train['lag_1'] * 7)).astype(int)
    classifier = _pipeline(RandomForestClassifier(n_estimators=60, min_samples_leaf=3, class_weight='balanced_subsample', random_state=42, n_jobs=2))
    classifier.fit(train[FEATURES], risk_target)
    risk_test = (test['lag_1'] + test['lag_7'] + test['lag_14'] + test['rolling_7'] * 4 >= np.maximum(1, test['lag_1'] * 7)).astype(int)
    class_pred = classifier.predict(test[FEATURES])
    try:
        f1 = float(f1_score(risk_test, class_pred, zero_division=0)); acc = float(accuracy_score(risk_test, class_pred))
    except Exception: f1, acc = 0.0, 0.0
    state['regressor'] = model; state['classifier'] = classifier
    state['metrics'] = {'mae': round(mae,3), 'rmse': round(rmse,3), 'wape_percent': round(wape,2), 'risk_accuracy': round(acc,3), 'risk_f1': round(f1,3), 'train_rows': int(len(train)), 'test_rows': int(len(test)), 'split_date': str(pd.Timestamp(split_date).date()), 'note': 'Chronological holdout. Risk classifier labels are simulated proxies unless real stockout labels are provided.'}
    state['history_source'] = source; state['history_rows'] = int(len(history))
    # Tree impurity feature importance is used for a lightweight explainability view.
    try:
        names = model.named_steps['preprocess'].get_feature_names_out()
        imps = model.named_steps['model'].feature_importances_
        grouped = {}
        for name, val in zip(names, imps):
            key = name.split('__', 1)[-1]
            if key.startswith(('Store_','Product_','Category_')): key = key.split('_',1)[0]
            grouped[key] = grouped.get(key, 0) + float(val)
        state['feature_importance'] = [{'feature': k, 'importance': round(v,4)} for k,v in sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:12]]
    except Exception: state['feature_importance'] = []
    joblib.dump(model, MODEL_DIR / 'demand_random_forest.joblib')
    joblib.dump(classifier, MODEL_DIR / 'stockout_risk_random_forest.joblib')
    return {'status':'trained','metrics':state['metrics'],'history_source':source,'history_rows':len(history),'usable_rows':len(engineered),'feature_importance':state['feature_importance']}

def _ensure_trained(inventory: pd.DataFrame | None = None):
    if state['regressor'] is not None: return
    if (MODEL_DIR/'demand_random_forest.joblib').exists() and (MODEL_DIR/'stockout_risk_random_forest.joblib').exists():
        try:
            state['regressor'] = joblib.load(MODEL_DIR/'demand_random_forest.joblib')
            state['classifier'] = joblib.load(MODEL_DIR/'stockout_risk_random_forest.joblib')
            if state['metrics'] is None: state['metrics'] = {'mae':None,'rmse':None,'wape_percent':None,'note':'Loaded existing model. Retrain to compute current holdout metrics.'}
            return
        except Exception: pass
    if inventory is None:
        if INVENTORY_PATH.exists(): inventory = pd.read_csv(INVENTORY_PATH)
        else: return
    hist, source = _ensure_history(inventory)
    _train(hist, source)

def _forecast_frame(inv: pd.DataFrame, horizon: int) -> list[dict]:
    _ensure_trained(inv)
    if state['regressor'] is None: raise HTTPException(503, 'Model is not trained. Call /api/train first.')
    # Build recent history grouped per store/product, then roll predictions forward.
    hist_path = HISTORY_PATH if HISTORY_PATH.exists() else SYNTHETIC_PATH
    if not hist_path.exists():
        hist, source = _ensure_history(inv)
        hist.to_csv(SYNTHETIC_PATH, index=False)
    else: hist = pd.read_csv(hist_path)
    hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
    hist['UnitsSold'] = pd.to_numeric(hist['UnitsSold'], errors='coerce').fillna(0)
    out=[]
    for _, item in inv.iterrows():
        series = hist[(hist['Store'].astype(str)==str(item['Store'])) & (hist['Product'].astype(str)==str(item['Product']))].sort_values('Date')
        vals = series['UnitsSold'].tail(30).astype(float).tolist()
        if not vals:
            vals = [max(1.0, float(item['Demand'])/7.0)] * 30
        last_date = series['Date'].max() if len(series) else pd.Timestamp.today().normalize()
        cat = str(item['Category']); store = str(item['Store']); product = str(item['Product']); price = float(item['Price'])
        daily=[]
        for step in range(1, horizon+1):
            date = pd.Timestamp(last_date) + pd.Timedelta(days=step)
            features = {'Store':store,'Product':product,'Category':cat,'Price':price,'Promotion':0,'day_of_week':date.dayofweek,'month':date.month,
                'lag_1': vals[-1], 'lag_7': vals[-7] if len(vals)>=7 else float(np.mean(vals)),
                'lag_14': vals[-14] if len(vals)>=14 else float(np.mean(vals)),
                'rolling_7': float(np.mean(vals[-7:])), 'rolling_14': float(np.mean(vals[-14:]))}
            x = pd.DataFrame([features], columns=FEATURES)
            pred = max(0.0, float(state['regressor'].predict(x)[0]))
            vals.append(pred); daily.append(pred)
        total = float(sum(daily)); avg = total / horizon
        # Classifier's probability is a learned proxy; risk score is combined with explicit stock coverage.
        final_date = pd.Timestamp(last_date) + pd.Timedelta(days=1)
        risk_features = {'Store':store,'Product':product,'Category':cat,'Price':price,'Promotion':0,'day_of_week':final_date.dayofweek,'month':final_date.month,
            'lag_1': vals[-horizon-1], 'lag_7': vals[max(0,len(vals)-horizon-7)], 'lag_14': vals[max(0,len(vals)-horizon-14)],
            'rolling_7': float(np.mean(vals[max(0,len(vals)-horizon-7):len(vals)-horizon])), 'rolling_14': float(np.mean(vals[max(0,len(vals)-horizon-14):len(vals)-horizon]))}
        prob = float(state['classifier'].predict_proba(pd.DataFrame([risk_features], columns=FEATURES))[0][1]) if state['classifier'] is not None else 0.0
        coverage = float(item['CurrentStock']) / max(1.0, total)
        # Coverage is a transparent business signal; do not present this as calibrated real-world probability.
        risk_score = min(0.99, max(0.01, 0.55 * prob + 0.45 * (1 - min(1.0, coverage))))
        risk = 'High' if risk_score >= 0.67 else ('Medium' if risk_score >= 0.35 else 'Low')
        out.append({'Store':store,'Product':product,'Category':cat,'CurrentStock':float(item['CurrentStock']), 'ForecastDailyAverage':round(avg,2), 'ForecastHorizonDemand':round(total,2), 'HorizonDays':horizon,
            'RecommendedStock':int(math.ceil(total*1.15)), 'ExpectedShortage':round(max(0,total-float(item['CurrentStock'])),2), 'CoverageRatio':round(coverage,2), 'StockoutRiskScore':round(risk_score,3), 'StockoutRisk':risk,
            'Explanation':f'{horizon}-day forecast is {total:.1f} units; available stock covers {coverage:.0%} of forecast demand. Risk combines model signal and transparent stock coverage.'})
    return sorted(out, key=lambda x: x['StockoutRiskScore'], reverse=True)

def _optimize(inv: pd.DataFrame, forecasts: list[dict]) -> dict:
    # Min-cost flow LP: each destination's shortage can be met by transfers or unmet-demand slack.
    f_by = {(x['Store'],x['Product']): x for x in forecasts}
    rows=[]
    for _, r in inv.iterrows():
        f=f_by.get((str(r['Store']),str(r['Product'])), {})
        demand=max(float(r['Demand']), float(f.get('ForecastHorizonDemand',r['Demand'])))
        rows.append({**r.to_dict(), 'target_demand':demand, 'excess':max(0,float(r['CurrentStock'])-demand*1.05), 'shortage':max(0,demand-float(r['CurrentStock']))})
    data=pd.DataFrame(rows)
    recs=[]; all_products=sorted(data['Product'].unique())
    for product in all_products:
        p=data[data['Product']==product].reset_index(drop=True)
        sources=[(i,r) for i,r in p.iterrows() if r['excess']>=1]
        dests=[(i,r) for i,r in p.iterrows() if r['shortage']>=1]
        if not sources or not dests: continue
        lanes=[]
        for si,s in sources:
            for di,d in dests:
                if s['Store']==d['Store']: continue
                fixed_lane_cost=max(0.0,(float(s['TransferCost'])+float(d['TransferCost']))/2)
                capacity=min(float(s['excess']),float(d['shortage']))
                # The supplied values (typically hundreds) are treated as shipment-level costs,
                # amortized over the maximum feasible lane quantity for the linear program.
                unit_cost=fixed_lane_cost/max(1.0,capacity)
                lanes.append((si,di,unit_cost,capacity,fixed_lane_cost))
        if not lanes: continue
        n=len(lanes); m=len(dests)
        # Unmet-demand slack is priced at the destination product price (lost-sales proxy),
        # so the optimizer will not recommend a transfer whose unit cost exceeds its estimated value.
        unmet_penalties=[max(0.0, float(d['Price'])) for _, d in dests]
        c=np.array([lane[2] for lane in lanes]+unmet_penalties, dtype=float)
        A_ub=[]; b_ub=[]
        for si,s in sources:
            A_ub.append([1.0 if lane[0]==si else 0.0 for lane in lanes]+[0.0]*m); b_ub.append(float(s['excess']))
        A_eq=[]; b_eq=[]
        for dj,d in dests:
            A_eq.append([1.0 if lane[1]==dj else 0.0 for lane in lanes]+[1.0 if j==list(idx for idx,_ in dests).index(dj) else 0.0 for j in range(m)])
            b_eq.append(float(d['shortage']))
        bounds=[(0, lane[3]) for lane in lanes]+[(0,None)]*m
        result=linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method='highs')
        if not result.success: continue
        for j,lane in enumerate(lanes):
            qty=int(math.floor(result.x[j]+1e-6))
            if qty<1: continue
            si,di,unit_cost,_,_=lane; s=p.iloc[si]; d=p.iloc[di]
            total_cost=round(lane[4],2)
            avoided=round(qty*float(d['Price']),2)
            recs.append({'id':f'LP-{len(recs)+1:03d}','Product':product,'Category':str(s['Category']),'StoreFrom':str(s['Store']),'StoreTo':str(d['Store']),
                'TransferQuantity':qty,'Priority':'High' if d['shortage']>=d['target_demand']*0.5 else 'Medium','UnitTransferCost':round(unit_cost,2),'TransferCost':total_cost,
                'EstimatedRevenueProtected':avoided,'EstimatedNetBenefit':round(avoided-total_cost,2),'UnmetDemandAfterTransfer':round(max(0,float(d['shortage'])-qty),2),
                'Reason':f'Linear-programming min-cost flow: {s["Store"]} has {s["excess"]:.0f} estimated excess units; {d["Store"]} has {d["shortage"]:.0f} forecast shortage. This lane costs about ₹{unit_cost:.2f} per unit.', 'Status':'Pending'})
    recs.sort(key=lambda x:(x['Priority']!='High', -x['EstimatedNetBenefit']))
    return {'algorithm':'Linear programming (SciPy HiGHS min-cost flow)', 'recommendations':recs, 'transfer_count':len(recs), 'total_units':sum(r['TransferQuantity'] for r in recs), 'estimated_transfer_cost':round(sum(r['TransferCost'] for r in recs),2), 'estimated_revenue_protected':round(sum(r['EstimatedRevenueProtected'] for r in recs),2), 'estimated_net_benefit':round(sum(r['EstimatedNetBenefit'] for r in recs),2), 'caveat':'TransferCost is interpreted as a shipment-level fixed cost and amortized over feasible lane capacity during optimization. Estimates use synthetic or user-provided forecasts; validate cost semantics before operational use.'}

@app.get('/api/health')
def health():
    return {'status':'ok','service':'NetworkIQ Intelligence API','model_trained':state['regressor'] is not None,'history_source':state['history_source'],'history_rows':state['history_rows']}

@app.post('/api/train')
async def train_model(file: UploadFile | None = File(default=None)):
    if file is not None:
        try: hist=pd.read_csv(file.file)
        except Exception as e: raise HTTPException(400, f'Could not read CSV: {e}')
        hist.to_csv(HISTORY_PATH, index=False); source='uploaded_or_user_provided'
    else:
        inventory = pd.read_csv(INVENTORY_PATH) if INVENTORY_PATH.exists() else pd.DataFrame()
        if inventory.empty: raise HTTPException(400, 'Provide a sales-history CSV or include inventory_data_large.csv in the project root.')
        if HISTORY_PATH.exists(): hist=pd.read_csv(HISTORY_PATH); source='uploaded_or_user_provided'
        elif SYNTHETIC_PATH.exists(): hist=pd.read_csv(SYNTHETIC_PATH); source='synthetic_demo'
        else:
            from generate_demo_data import generate
            hist=generate(INVENTORY_PATH,SYNTHETIC_PATH,180,42); source='synthetic_demo'
    return _train(hist,source)

@app.post('/api/train-demo')
def train_demo_model():
    inventory = pd.read_csv(INVENTORY_PATH) if INVENTORY_PATH.exists() else pd.DataFrame()
    if inventory.empty:
        raise HTTPException(400, 'Include inventory_data_large.csv in the project root or upload sales history.')
    if HISTORY_PATH.exists():
        hist = pd.read_csv(HISTORY_PATH); source = 'uploaded_or_user_provided'
    elif SYNTHETIC_PATH.exists():
        hist = pd.read_csv(SYNTHETIC_PATH); source = 'synthetic_demo'
    else:
        from generate_demo_data import generate
        hist = generate(INVENTORY_PATH, SYNTHETIC_PATH, 180, 42); source = 'synthetic_demo'
    return _train(hist, source)

@app.post('/api/forecast')
def forecast(req: InventoryRequest):
    inv=_inventory_df(req.inventory)
    forecasts=_forecast_frame(inv, req.horizon_days)
    return {'model':'RandomForestRegressor','horizon_days':req.horizon_days,'history_source':state['history_source'],'metrics':state['metrics'],'forecasts':forecasts,'feature_importance':state['feature_importance'],'synthetic_data_warning':state['history_source']=='synthetic_demo'}

@app.post('/api/optimize')
def optimize(req: InventoryRequest):
    inv=_inventory_df(req.inventory)
    forecasts=_forecast_frame(inv,req.horizon_days)
    result=_optimize(inv,forecasts)
    result['forecasts']=forecasts
    result['history_source']=state['history_source']
    return result

@app.post('/api/assistant')
def assistant(req: AssistantRequest):
    inv=_inventory_df(req.inventory) if req.inventory else pd.DataFrame()
    question=req.question.strip()
    api_key=os.getenv('OPENAI_API_KEY','').strip()
    if api_key:
        try:
            from openai import OpenAI
            client=OpenAI(api_key=api_key)
            context={'inventory_rows':len(inv),'inventory_summary':{},'forecasts':req.forecasts[:30],'transfer_recommendations':req.transfers[:20]}
            if not inv.empty:
                context['inventory_summary']={'stores':int(inv['Store'].nunique()),'products':int(inv['Product'].nunique()),'units':float(inv['CurrentStock'].sum()),'inventory_value':float((inv['CurrentStock']*inv['Price']).sum()),'low_coverage_rows':int((inv['CurrentStock']<inv['Demand']).sum())}
            response=client.chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'),temperature=0.2,messages=[
                {'role':'system','content':'You are NetworkIQ, an inventory analytics assistant. Answer only from the supplied inventory/forecast/transfer context. Clearly state when data is missing. Treat forecast values and financial outcomes as estimates, never invent metrics. Keep answers concise and actionable.'},
                {'role':'user','content':f'Question: {question}\nGrounding data (JSON): {json.dumps(context,default=str)[:24000]}'}])
            return {'answer':response.choices[0].message.content or 'I could not generate a response.','mode':'llm','model':os.getenv('OPENAI_MODEL','gpt-4o-mini')}
        except Exception as e:
            # Safe graceful fallback if API key is invalid or service unavailable.
            fallback_error=str(e)[:200]
    else: fallback_error=None
    q=question.lower(); answer=''
    if inv.empty:
        answer='Load or analyze an inventory dataset first so I can ground the answer in your actual records. You can also configure OPENAI_API_KEY for conversational answers.'
    elif any(k in q for k in ['critical','stockout','shortage','risk','run out']):
        if req.forecasts:
            high=[f for f in req.forecasts if f.get('StockoutRisk')=='High']
            answer=f'I found {len(high)} high-risk product/store combinations among {len(req.forecasts)} forecast rows. '
            answer+='Top risks: '+ '; '.join(f"{f.get('Product')} at {f.get('Store')} (risk {float(f.get('StockoutRiskScore',0)):.0%}, forecast {f.get('ForecastHorizonDemand')} units vs {f.get('CurrentStock')} in stock)" for f in high[:5]) if high else 'No rows currently meet the High risk threshold in the supplied forecasts.'
        else: answer=f'{int((inv.CurrentStock<inv.Demand).sum())} inventory rows currently have stock below the Demand column. Generate a forecast to calculate forward-looking risk.'
    elif any(k in q for k in ['transfer','move','recommend']):
        answer=f'There are {len(req.transfers)} transfer recommendations in the supplied optimization result. '
        answer+='; '.join(f"move {t.get('TransferQuantity')} {t.get('Product')} from {t.get('StoreFrom')} to {t.get('StoreTo')}" for t in req.transfers[:5]) or 'Run the transfer optimizer to generate a cost-aware recommendation list.'
    elif any(k in q for k in ['value','inventory worth','units','stock']):
        answer=f"The loaded dataset contains {len(inv)} inventory rows across {inv.Store.nunique()} stores and {inv.Product.nunique()} products, with {inv.CurrentStock.sum():,.0f} units and estimated inventory value ₹{(inv.CurrentStock*inv.Price).sum():,.2f}."
    elif any(k in q for k in ['forecast','demand','predict']):
        if req.forecasts:
            top=sorted(req.forecasts,key=lambda x:float(x.get('ForecastHorizonDemand',0)),reverse=True)[:5]
            answer='Highest forecast demand rows: '+'; '.join(f"{f.get('Product')} at {f.get('Store')}: {f.get('ForecastHorizonDemand')} units over {f.get('HorizonDays')} days" for f in top)
        else: answer='Run demand forecasting first. NetworkIQ uses a Random Forest model with lagged sales, rolling averages, calendar features, product, store, category, price and promotion.'
    else:
        answer=f"I can help with demand forecasts, stockout risks, inventory value and transfer recommendations. Current dataset: {len(inv)} rows, {inv.Store.nunique()} stores, {inv.Product.nunique()} products. Ask a focused question or run Forecast/Optimize first."
    if fallback_error: answer+=f' (LLM unavailable; used local grounded fallback: {fallback_error})'
    return {'answer':answer,'mode':'local_grounded','model':None}

@app.post('/api/upload-history')
async def upload_history(file: UploadFile = File(...)):
    try: hist=pd.read_csv(file.file)
    except Exception as e: raise HTTPException(400,f'Could not read CSV: {e}')
    required={'Date','Store','Product','UnitsSold'}
    if not required.issubset(hist.columns): raise HTTPException(400,'Required columns: Date, Store, Product, UnitsSold. Optional: Category, Price, Promotion.')
    hist.to_csv(HISTORY_PATH,index=False)
    state['regressor']=None; state['classifier']=None
    result=_train(hist,'uploaded_or_user_provided')
    return {'message':'Sales history uploaded and model trained.','training':result}

# Best-effort startup: train on included synthetic history so the API works on first launch.
@app.on_event('startup')
def startup():
    try:
        if INVENTORY_PATH.exists():
            inv=pd.read_csv(INVENTORY_PATH)
            _ensure_trained(inv)
    except Exception as exc:
        print(f'[NetworkIQ] Startup training deferred: {exc}')
