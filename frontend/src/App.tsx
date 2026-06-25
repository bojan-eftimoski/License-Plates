import { useEffect, useState } from 'react'
import './App.css'
import Upload from './Upload'
import Camera from './Camera'
import { makeT, type Lang } from './i18n'

const DEFAULT_API = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'
type Status = 'connecting' | 'ready' | 'offline'

const storedLang = localStorage.getItem('lang')
const INITIAL_LANG: Lang = storedLang === 'en' || storedLang === 'mk' ? storedLang : 'mk'

export default function App() {
  const [apiBase, setApiBase] = useState(localStorage.getItem('apiBase') || DEFAULT_API)
  const [mode, setMode] = useState<'upload' | 'camera'>('upload')
  const [status, setStatus] = useState<Status>('connecting')
  const [lang, setLang] = useState<Lang>(INITIAL_LANG)
  const base = apiBase.replace(/\/+$/, '')
  const t = makeT(lang)

  function changeLang(l: Lang) {
    setLang(l)
    localStorage.setItem('lang', l)
  }

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

  const statusText = status === 'ready' ? t('status_ready') : status === 'connecting' ? t('status_connecting') : t('status_offline')

  return (
    <div className="app">
      <header>
        <h1>{t('app_title')}</h1>
        <p className="sub">{t('app_sub')}</p>
        <div className="langtoggle">
          <button className={lang === 'en' ? 'on' : ''} onClick={() => changeLang('en')}>EN</button>
          <button className={lang === 'mk' ? 'on' : ''} onClick={() => changeLang('mk')}>MK</button>
        </div>
      </header>

      <div className="tabs">
        <button className={mode === 'upload' ? 'on' : ''} onClick={() => setMode('upload')}>{t('tab_upload')}</button>
        <button className={mode === 'camera' ? 'on' : ''} onClick={() => setMode('camera')}>{t('tab_camera')}</button>
      </div>

      <section className="card">
        <label className="field">
          <span>
            {t('backend_url')} <em className={`status ${status}`}>● {statusText}</em>
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

      {mode === 'upload' ? <Upload base={base} t={t} /> : <Camera base={base} t={t} />}

      <footer>{t('footer')}</footer>
    </div>
  )
}
