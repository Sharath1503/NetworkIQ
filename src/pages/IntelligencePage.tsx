import { useMemo, useState } from 'react';
import { AlertTriangle, BrainCircuit, CheckCircle2, CircleHelp, FileUp, Loader2, MessageCircle, RefreshCw, Send, Sparkles, TrendingUp, Truck } from 'lucide-react';
import { useApp } from '@/context/AppContext';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';

type Forecast = {
  Store: string; Product: string; Category: string; CurrentStock: number; ForecastDailyAverage: number;
  ForecastHorizonDemand: number; HorizonDays: number; RecommendedStock: number; ExpectedShortage: number;
  CoverageRatio: number; StockoutRiskScore: number; StockoutRisk: 'High'|'Medium'|'Low'; Explanation: string;
};
type Transfer = { id: string; Product: string; StoreFrom: string; StoreTo: string; TransferQuantity: number; Priority: string; TransferCost: number; EstimatedRevenueProtected: number; EstimatedNetBenefit: number; Reason: string };
type ApiResult = { metrics?: Record<string, unknown>; forecasts?: Forecast[]; feature_importance?: {feature:string;importance:number}[]; history_source?: string; history_rows?: number; synthetic_data_warning?: boolean; recommendations?: Transfer[]; transfer_count?: number; total_units?: number; estimated_transfer_cost?: number; estimated_revenue_protected?: number; estimated_net_benefit?: number; algorithm?: string; caveat?: string };
const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export function IntelligencePage() {
  const { dataset } = useApp();
  const [horizon, setHorizon] = useState('7');
  const [loading, setLoading] = useState<'train'|'forecast'|'optimize'|'history'|null>(null);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [error, setError] = useState('');
  const [question, setQuestion] = useState('Which products have the highest stockout risk?');
  const [chat, setChat] = useState<{q:string;a:string;mode?:string}[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [historyFile, setHistoryFile] = useState<File | null>(null);
  const inventory = useMemo(() => dataset.map((r) => ({...r})), [dataset]);
  const request = async (endpoint: string, body?: unknown): Promise<any> => {
    const response = await fetch(`${API}${endpoint}`, body instanceof FormData ? {method:'POST',body} : body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status}). Is the NetworkIQ backend running?`);
    return data;
  };
  const runTrain = async () => {
    setLoading('train'); setError('');
    try { const data = await request('/api/train-demo', {}); setResult((old) => ({...old,...data})); }
    catch (e) { setError(e instanceof Error ? e.message : 'Training failed.'); }
    finally { setLoading(null); }
  };
  const runForecast = async () => {
    if (!inventory.length) { setError('Load your inventory CSV or sample dataset in Upload Dataset first.'); return; }
    setLoading('forecast'); setError('');
    try { const data = await request('/api/forecast',{inventory,horizon_days:Number(horizon)}); setResult((old) => ({...old,...data})); }
    catch (e) { setError(e instanceof Error ? e.message : 'Forecast failed.'); }
    finally { setLoading(null); }
  };
  const runOptimize = async () => {
    if (!inventory.length) { setError('Load your inventory CSV or sample dataset in Upload Dataset first.'); return; }
    setLoading('optimize'); setError('');
    try { const data = await request('/api/optimize',{inventory,horizon_days:Number(horizon)}); setResult((old) => ({...old,...data})); }
    catch (e) { setError(e instanceof Error ? e.message : 'Optimization failed.'); }
    finally { setLoading(null); }
  };
  const uploadHistory = async () => {
    if (!historyFile) { setError('Choose a historical sales CSV first.'); return; }
    const form = new FormData(); form.append('file', historyFile); setLoading('history'); setError('');
    try { const data = await request('/api/upload-history',form); setResult((old) => ({...old,...data.training})); }
    catch (e) { setError(e instanceof Error ? e.message : 'History upload failed.'); }
    finally { setLoading(null); }
  };
  const ask = async () => {
    if (!question.trim()) return;
    const q=question.trim(); setChatLoading(true); setQuestion('');
    try { const data=await request('/api/assistant',{question:q,inventory,forecasts:result?.forecasts||[],transfers:result?.recommendations||[]}); setChat((prev)=>[...prev,{q,a:data.answer,mode:data.mode}]); }
    catch(e) { setChat((prev)=>[...prev,{q,a:e instanceof Error?e.message:'Assistant request failed.'}]); }
    finally { setChatLoading(false); }
  };
  const busy = loading !== null;
  const money = (n?: number) => `₹${Number(n||0).toLocaleString('en-IN',{maximumFractionDigits:2})}`;

  return <div className="space-y-6">
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div><div className="mb-2 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-3 py-1 text-xs font-semibold text-primary"><Sparkles className="h-3.5 w-3.5"/> ML WORKBENCH</div><h1 className="text-2xl font-bold tracking-tight">NetworkIQ Intelligence</h1><p className="mt-1 text-sm text-muted-foreground">Train demand forecasting, estimate stockout risk, optimize transfers, and ask inventory questions.</p></div>
      <Badge variant="outline" className="w-fit">{inventory.length} inventory rows loaded</Badge>
    </div>

    {error && <div className="flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0"/><span>{error}</span></div>}
    {result?.synthetic_data_warning && <div className="flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 text-sm"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600"/><span><b>Synthetic demo data:</b> forecasts currently use generated sales history, not real retailer transactions. Upload dated sales data to evaluate on your real distribution.</span></div>}

    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <Card><CardHeader className="pb-2"><CardDescription>Demand Forecasting</CardDescription><CardTitle className="flex items-center gap-2 text-lg"><BrainCircuit className="h-5 w-5 text-primary"/>Random Forest</CardTitle></CardHeader><CardContent><p className="text-xs text-muted-foreground">Lagged sales, rolling averages, calendar, price, promotion and store/product features.</p><Button className="mt-4 w-full" onClick={runTrain} disabled={busy}>{loading==='train'?<Loader2 className="mr-2 h-4 w-4 animate-spin"/>:<RefreshCw className="mr-2 h-4 w-4"/>}{loading==='train'?'Training model':'Train / retrain model'}</Button></CardContent></Card>
      <Card><CardHeader className="pb-2"><CardDescription>Forecast horizon</CardDescription><CardTitle className="text-lg">Future demand</CardTitle></CardHeader><CardContent><label className="mb-2 block text-xs text-muted-foreground">Days to forecast</label><select value={horizon} onChange={e=>setHorizon(e.target.value)} className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"><option value="7">Next 7 days</option><option value="14">Next 14 days</option><option value="30">Next 30 days</option></select><Button className="mt-4 w-full" onClick={runForecast} disabled={busy}>{loading==='forecast'?<Loader2 className="mr-2 h-4 w-4 animate-spin"/>:<TrendingUp className="mr-2 h-4 w-4"/>}Forecast demand</Button></CardContent></Card>
      <Card><CardHeader className="pb-2"><CardDescription>Transfer optimization</CardDescription><CardTitle className="flex items-center gap-2 text-lg"><Truck className="h-5 w-5 text-primary"/>Linear programming</CardTitle></CardHeader><CardContent><p className="text-xs text-muted-foreground">Min-cost flow across stores while respecting estimated supply and destination shortages.</p><Button className="mt-4 w-full" onClick={runOptimize} disabled={busy}>{loading==='optimize'?<Loader2 className="mr-2 h-4 w-4 animate-spin"/>:<CheckCircle2 className="mr-2 h-4 w-4"/>}Optimize transfers</Button></CardContent></Card>
      <Card><CardHeader className="pb-2"><CardDescription>Use your own sales history</CardDescription><CardTitle className="text-lg">Upload CSV</CardTitle></CardHeader><CardContent><Input type="file" accept=".csv,text/csv" className="text-xs" onChange={e=>setHistoryFile(e.target.files?.[0]||null)}/><p className="mt-2 text-[11px] text-muted-foreground">Required: Date, Store, Product, UnitsSold. Optional: Category, Price, Promotion.</p><Button className="mt-3 w-full" variant="outline" onClick={uploadHistory} disabled={busy||!historyFile}>{loading==='history'?<Loader2 className="mr-2 h-4 w-4 animate-spin"/>:<FileUp className="mr-2 h-4 w-4"/>}Upload & train</Button></CardContent></Card>
    </div>

    {result?.metrics && <Card><CardHeader><CardTitle className="flex items-center gap-2"><TrendingUp className="h-5 w-5 text-primary"/>Model evaluation</CardTitle><CardDescription>Chronological holdout evaluation; results are estimates and depend on the selected dataset.</CardDescription></CardHeader><CardContent><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">{[['MAE',result.metrics.mae],['RMSE',result.metrics.rmse],['WAPE',result.metrics.wape_percent==null?null:`${result.metrics.wape_percent}%`],['Train rows',result.metrics.train_rows],['Test rows',result.metrics.test_rows]].map(([label,value])=><div key={String(label)} className="rounded-xl bg-muted/50 p-3"><p className="text-xs text-muted-foreground">{String(label)}</p><p className="mt-1 text-xl font-bold">{value == null ? '—' : `${value}`} </p></div>)}</div><p className="mt-3 text-xs text-muted-foreground">{String(result.metrics.note||'Evaluated using a time-ordered train/test split.')}</p></CardContent></Card>}

    {result?.forecasts && <Card><CardHeader><CardTitle>Demand forecasts & stockout risk</CardTitle><CardDescription>Sorted by highest estimated risk. Forecast demand is cumulative across the selected horizon.</CardDescription></CardHeader><CardContent><div className="overflow-x-auto"><table className="w-full min-w-[760px] text-left text-sm"><thead><tr className="border-b text-xs text-muted-foreground"><th className="p-2">Product / Store</th><th className="p-2">Stock</th><th className="p-2">Forecast</th><th className="p-2">Recommended stock</th><th className="p-2">Coverage</th><th className="p-2">Risk</th></tr></thead><tbody>{result.forecasts.slice(0,80).map((f,i)=><tr key={`${f.Store}-${f.Product}-${i}`} className="border-b border-border/60"><td className="p-2"><div className="font-medium">{f.Product}</div><div className="text-xs text-muted-foreground">{f.Store}</div></td><td className="p-2">{f.CurrentStock}</td><td className="p-2">{f.ForecastHorizonDemand}</td><td className="p-2">{f.RecommendedStock}</td><td className="p-2">{Math.round(f.CoverageRatio*100)}%</td><td className="p-2"><Badge variant={f.StockoutRisk==='High'?'destructive':'outline'}>{f.StockoutRisk} · {Math.round(f.StockoutRiskScore*100)}%</Badge></td></tr>)}</tbody></table></div>{result.forecasts.length>80&&<p className="mt-2 text-xs text-muted-foreground">Showing first 80 of {result.forecasts.length} records.</p>}</CardContent></Card>}

    <div className="grid gap-4 xl:grid-cols-2">
      {result?.feature_importance && result.feature_importance.length>0 && <Card><CardHeader><CardTitle>Model feature importance</CardTitle><CardDescription>Random Forest feature importance aggregated by feature group; not a causal explanation.</CardDescription></CardHeader><CardContent className="space-y-3">{result.feature_importance.map((f)=><div key={f.feature}><div className="mb-1 flex justify-between text-sm"><span>{f.feature}</span><span className="text-muted-foreground">{(f.importance*100).toFixed(1)}%</span></div><div className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-primary" style={{width:`${Math.min(100,f.importance*100)}%`}}/></div></div>)}</CardContent></Card>}
      {result?.recommendations && <Card><CardHeader><CardTitle className="flex items-center gap-2"><Truck className="h-5 w-5 text-primary"/>Optimized transfer plan</CardTitle><CardDescription>{result.algorithm} · {result.transfer_count} transfer lanes recommended</CardDescription></CardHeader><CardContent><div className="mb-4 grid grid-cols-2 gap-3"><div className="rounded-xl bg-muted/50 p-3"><p className="text-xs text-muted-foreground">Transfer cost</p><p className="text-lg font-bold">{money(result.estimated_transfer_cost)}</p></div><div className="rounded-xl bg-muted/50 p-3"><p className="text-xs text-muted-foreground">Estimated net benefit</p><p className="text-lg font-bold">{money(result.estimated_net_benefit)}</p></div></div><div className="space-y-3">{result.recommendations.slice(0,15).map((r)=><div key={r.id} className="rounded-xl border p-3"><div className="flex flex-wrap items-center justify-between gap-2"><b>{r.Product} · {r.TransferQuantity} units</b><Badge variant={r.Priority==='High'?'destructive':'outline'}>{r.Priority}</Badge></div><p className="mt-1 text-sm text-muted-foreground">{r.StoreFrom} → {r.StoreTo}</p><p className="mt-1 text-xs">Transfer cost {money(r.TransferCost)} · Revenue protected {money(r.EstimatedRevenueProtected)} · Net benefit {money(r.EstimatedNetBenefit)}</p></div>)}</div>{result.caveat&&<p className="mt-3 text-xs text-muted-foreground">{result.caveat}</p>}</CardContent></Card>}
    </div>

    <Card><CardHeader><CardTitle className="flex items-center gap-2"><MessageCircle className="h-5 w-5 text-primary"/>NetworkIQ inventory assistant</CardTitle><CardDescription>Answers are grounded in the loaded inventory, forecasts, and transfer results. Add an API key to enable an LLM; otherwise, the local grounded assistant still works.</CardDescription></CardHeader><CardContent><div className="mb-4 space-y-3">{chat.length===0?<div className="flex items-start gap-2 rounded-xl bg-muted/50 p-4 text-sm"><CircleHelp className="mt-0.5 h-4 w-4 text-primary"/><p>Try: “Which products are at highest risk?”, “What transfers should we make?”, or “What is our inventory value?”</p></div>:chat.map((c,i)=><div key={i} className="space-y-2 rounded-xl border p-3"><p className="text-sm font-semibold">You: {c.q}</p><p className="whitespace-pre-wrap text-sm text-muted-foreground">{c.a}</p><span className="text-[10px] uppercase tracking-wide text-muted-foreground">{c.mode||'assistant'}</span></div>)}</div><form onSubmit={e=>{e.preventDefault();void ask();}} className="flex gap-2"><Input value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Ask about forecasts, stockouts, or transfers…"/><Button type="submit" disabled={chatLoading||!question.trim()}>{chatLoading?<Loader2 className="h-4 w-4 animate-spin"/>:<Send className="h-4 w-4"/>}</Button></form></CardContent></Card>

    <div className="rounded-xl border border-primary/15 bg-primary/5 p-4 text-xs leading-relaxed text-muted-foreground"><b className="text-foreground">Research / demo transparency:</b> The included dataset is synthetic and is clearly marked as such. The model metrics are computed using a chronological holdout. The stockout-risk classifier currently uses proxy labels generated from sales patterns; replace these with real historical stockout outcomes before claiming calibrated stockout probabilities. Transfer savings are estimates, not guaranteed financial outcomes. Configure <code>OPENAI_API_KEY</code> to enable the optional LLM assistant.</div>
  </div>;
}
