import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { PlaygroundView } from './components/PlaygroundView';
import { BenchmarksView } from './components/BenchmarksView';
import { ArchitectureView } from './components/ArchitectureView';
import { HistoryView } from './components/HistoryView';
import { fetchHealth, fetchSystemInfo, fetchEngines } from './lib/api';
import type { SystemInfo, EngineInfo } from './types';

export function App() {
  const [activeTab, setActiveTab] = useState<'playground' | 'benchmarks' | 'architecture' | 'history'>('playground');
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [serverOnline, setServerOnline] = useState<boolean>(false);

  const checkStatus = async () => {
    try {
      await fetchHealth();
      setServerOnline(true);
      const sys = await fetchSystemInfo();
      setSystemInfo(sys);
    } catch (e) {
      setServerOnline(false);
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 5000);

    fetchEngines()
      .then((res) => {
        if (res && res.engines) setEngines(res.engines);
      })
      .catch((e) => console.warn('Engines lookup warning:', e));

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-obsidian-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-black">
      {/* Top Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        systemInfo={systemInfo}
        serverOnline={serverOnline}
      />

      {/* Main Content Area */}
      <main className="flex-1">
        {activeTab === 'playground' && (
          <PlaygroundView engines={engines} serverOnline={serverOnline} />
        )}
        {activeTab === 'benchmarks' && <BenchmarksView serverOnline={serverOnline} />}
        {activeTab === 'architecture' && <ArchitectureView />}
        {activeTab === 'history' && <HistoryView />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-obsidian-900/60 py-6 mt-12 text-xs font-mono text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="text-emerald-400 font-bold">LIEFS</span>
            <span>• LLM Inference Engine From Scratch</span>
            <span>• Qwen2.5-0.5B-Instruct</span>
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>KV-Cache</span>
            <span>•</span>
            <span>Continuous Batching</span>
            <span>•</span>
            <span>Paged Attention</span>
            <span>•</span>
            <span>INT8 Quantization</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
