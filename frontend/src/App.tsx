import { useEffect, useState } from 'react'
import './App.css'
import Upload from './Upload'
import Camera from './Camera'

const DEFAULT_API = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'
type Status = 'connecting' | 'ready' | 'offline'

export default function App() {
  const [apiBase, setApiBase] = useState(localStorage.getItem('apiBase') || DEFAULT_API)
  const [mode, setMode] = useState<'upload' | 'camera'>('upload')
  const [status, setStatus] = useState<Status>('connecting')
  const base = apiBase.replace(/\/+$/, '')

  // Ping health on load / URL change: wakes a sleeping HF Space and shows backend status.
  useEffect(() => {
    setStatus('connecting')
    const ctrl = new AbortController()
    const t = setTimeout(() => ctrl.abort(), 45000) // first wake after sleep can take ~30s
    fetch(`${base}/api/health`, { signal: ctrl.signal })
      .then((r) => setStatus(r.ok ? 'ready' : 'offline'))
      .catch(() => setStatus('offline'))
      .finally(() => clearTimeout(t))
    return () => {
      clearTimeout(t)
      ctrl.abort()
    }
  }, [base])

  const statusText = status === 'ready' ? 'backend ready' : status === 'connecting' ? 'waking backend…' : 'backend offline'

  return (
    <div className="app">
      <header>
        <h1>🚗 Macedonian Plate Reader</h1>
        <p className="sub">Classical computer vision · detect &amp; read MK license plates</p>
      </header>

      <div className="tabs">
        <button className={mode === 'upload' ? 'on' : ''} onClick={() => setMode('upload')}>Upload</button>
        <button className={mode === 'camera' ? 'on' : ''} onClick={() => setMode('camera')}>Live camera</button>
      </div>

      <section className="card">
        <label className="field">
          <span>
            Backend URL <em className={`status ${status}`}>● {statusText}</em>
          </span>
          <input
            value={apiBase}
            spellCheck={false}
            onChange={(e) => {
              setApiBase(e.target.value)
              localStorage.setItem('apiBase', e.target.value)
            }}
          />
        </label>
      </section>

      {mode === 'upload' ? <Upload base={base} /> : <Camera base={base} />}

      <footer>Detect-always, read-when-confident · graded model runs on the backend (cv2 + numpy, no deep learning)</footer>
    </div>
  )
}
