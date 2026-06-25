import { useState } from 'react'
import './App.css'
import Upload from './Upload'
import Camera from './Camera'

const DEFAULT_API = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

export default function App() {
  const [apiBase, setApiBase] = useState(localStorage.getItem('apiBase') || DEFAULT_API)
  const [mode, setMode] = useState<'upload' | 'camera'>('upload')
  const base = apiBase.replace(/\/+$/, '')

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
          <span>Backend URL</span>
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
