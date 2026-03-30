import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Shield, Brain, TrendingUp, Briefcase, Loader2, CheckCircle } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { getRiskSettings, updateRiskSettings } from '../api';

export default function RiskSettingsPage() {
    const { user } = useAuth();
    const [maxDailyLoss, setMaxDailyLoss] = useState('5000');
    const [maxDailyTrades, setMaxDailyTrades] = useState('10');
    const [maxPositionPct, setMaxPositionPct] = useState('20');
    const [autoSL, setAutoSL] = useState(true);
    const [autoTarget, setAutoTarget] = useState(true);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);

    useEffect(() => {
        if (!user) return;
        getRiskSettings(user.id)
            .then((s) => {
                setMaxDailyLoss(String(s.max_daily_loss || 5000));
                setMaxDailyTrades(String(s.max_daily_trades || 10));
                setMaxPositionPct(String(s.max_position_pct || 20));
                setAutoSL(!!s.auto_stop_loss);
                setAutoTarget(!!s.auto_target);
            })
            .catch(() => { })
            .finally(() => setLoading(false));
    }, [user]);

    const handleSave = async () => {
        if (!user) return;
        setSaving(true); setSaved(false);
        try {
            await updateRiskSettings(user.id, {
                max_daily_loss: parseFloat(maxDailyLoss),
                max_daily_trades: parseInt(maxDailyTrades),
                max_position_pct: parseFloat(maxPositionPct),
                auto_stop_loss: autoSL ? 1 : 0,
                auto_target: autoTarget ? 1 : 0,
            });
            setSaved(true);
            setTimeout(() => setSaved(false), 3000);
        } catch { } finally { setSaving(false); }
    };

    if (loading) return <div className="flex items-center justify-center py-20 text-slate-400">Loading settings...</div>;

    return (
        <>
            <h1 className="text-white text-3xl md:text-4xl font-black tracking-tight">Risk Management</h1>
            <p className="text-slate-400 max-w-xl">Configure safety parameters. AI will halt trading or alert you if thresholds are breached.</p>

            {/* Capital Protection */}
            <div className="mt-6">
                <div className="flex items-center gap-2 mb-6"><Shield className="w-5 h-5 text-primary" /><h2 className="text-white text-xl font-bold">Capital Protection</h2></div>
                <div className="space-y-4">
                    <div className="bg-surface-dark border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-all">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center"><TrendingUp className="w-5 h-5 text-primary" /></div>
                                <div><p className="text-white font-bold">Max Daily Loss</p><p className="text-slate-400 text-sm">Halt all trading if daily loss exceeds this limit.</p></div>
                            </div>
                        </div>
                        <div className="relative"><span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">₹</span>
                            <input type="number" value={maxDailyLoss} onChange={(e) => setMaxDailyLoss(e.target.value)} className="w-full bg-slate-950/50 border border-slate-700 text-white rounded-xl p-3.5 pl-8 outline-none focus:ring-2 focus:ring-primary/50" />
                        </div>
                    </div>
                    <div className="bg-surface-dark border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-all">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center"><Briefcase className="w-5 h-5 text-primary" /></div>
                            <div><p className="text-white font-bold">Max Daily Trades</p><p className="text-slate-400 text-sm">Limit the number of trades per day.</p></div>
                        </div>
                        <input type="number" value={maxDailyTrades} onChange={(e) => setMaxDailyTrades(e.target.value)} className="w-full bg-slate-950/50 border border-slate-700 text-white rounded-xl p-3.5 outline-none focus:ring-2 focus:ring-primary/50" />
                    </div>
                    <div className="bg-surface-dark border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-all">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center"><Brain className="w-5 h-5 text-primary" /></div>
                            <div><p className="text-white font-bold">Max Position Size</p><p className="text-slate-400 text-sm">Maximum % of portfolio in a single position.</p></div>
                        </div>
                        <div className="relative">
                            <input type="number" value={maxPositionPct} onChange={(e) => setMaxPositionPct(e.target.value)} className="w-full bg-slate-950/50 border border-slate-700 text-white rounded-xl p-3.5 pr-8 outline-none focus:ring-2 focus:ring-primary/50" />
                            <span className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500">%</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Auto features */}
            <div className="mt-8">
                <div className="flex items-center gap-2 mb-6"><Brain className="w-5 h-5 text-primary" /><h2 className="text-white text-xl font-bold">Automation</h2></div>
                <div className="space-y-4">
                    <div className="bg-surface-dark border border-slate-800 rounded-xl p-6 flex items-center justify-between">
                        <div><p className="text-white font-bold">Auto Stop-Loss</p><p className="text-slate-400 text-sm">Automatically place SL orders on new trades.</p></div>
                        <button onClick={() => setAutoSL(!autoSL)} className={`w-12 h-6 rounded-full transition-colors relative ${autoSL ? 'bg-primary' : 'bg-slate-600'}`}>
                            <div className={`w-5 h-5 rounded-full bg-white absolute top-0.5 transition-transform ${autoSL ? 'translate-x-6' : 'translate-x-0.5'}`} />
                        </button>
                    </div>
                    <div className="bg-surface-dark border border-slate-800 rounded-xl p-6 flex items-center justify-between">
                        <div><p className="text-white font-bold">Auto Target</p><p className="text-slate-400 text-sm">Automatically place target exit orders on new trades.</p></div>
                        <button onClick={() => setAutoTarget(!autoTarget)} className={`w-12 h-6 rounded-full transition-colors relative ${autoTarget ? 'bg-primary' : 'bg-slate-600'}`}>
                            <div className={`w-5 h-5 rounded-full bg-white absolute top-0.5 transition-transform ${autoTarget ? 'translate-x-6' : 'translate-x-0.5'}`} />
                        </button>
                    </div>
                </div>
            </div>

            {/* Save */}
            <div className="flex justify-end gap-4 mt-8">
                <Link to="/dashboard" className="px-6 py-3 rounded-xl text-slate-400 text-sm font-medium hover:text-white transition-colors">Cancel</Link>
                <button onClick={handleSave} disabled={saving} className="px-8 py-3 rounded-xl bg-primary hover:bg-primary/90 text-white text-sm font-bold transition-colors shadow-lg shadow-primary/25 disabled:opacity-50 flex items-center gap-2">
                    {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : saved ? <CheckCircle className="w-4 h-4" /> : null}
                    {saving ? 'Saving...' : saved ? 'Saved!' : 'Save Risk Profile'}
                </button>
            </div>
        </>
    );
}
