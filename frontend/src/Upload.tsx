import { useEffect, useState } from 'react'
import type { T } from './i18n'

type Plate = { plate_text: string | null; region: string | null; confidence: number; frame?: number }
type AnalyzeResult = { type: 'image' | 'video'; plates: Plate[]; annotated?: string | null; frames?: number }

export default function Upload({ base, t }: { base: string; t: T }) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalyzeResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [copied, setCopied] = useState(false)

  function pick(f: File | null) {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
    setPreview(f.type.startsWith('image') ? URL.createObjectURL(f) : null)
  }

  // Paste an image from the clipboard anywhere on the page (Ctrl/Cmd+V or screenshot paste).
  useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.type.startsWith('image'))
      const f = item?.getAsFile()
      if (f) { e.preventDefault(); pick(new File([f], f.name || 'pasted.png', { type: f.type })) }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [])

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f && (f.type.startsWith('image') || f.type.startsWith('video'))) pick(f)
    else if (f) setError(t('drop_invalid'))
  }

  async function analyze() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const r = await fetch(`${base}/api/analyze`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(`server returned ${r.status}`)
      setResult((await r.json()) as AnalyzeResult)
    } catch (e) {
      setError(t('backend_unreachable', { base, err: (e as Error).message }))
    } finally {
      setLoading(false)
    }
  }

  const reads = result?.plates.filter((p) => p.plate_text) ?? []

  async function copyCsv() {
    const rows = [['plate', 'region', 'confidence'], ...reads.map((p) =>
      [p.plate_text ?? '', p.region ?? '', `${Math.round(p.confidence * 100)}%`])]
    const csv = rows.map((r) => r.join(',')).join('\n')
    try {
      await navigator.clipboard.writeText(csv)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setError(t('clipboard_blocked'))
    }
  }

  return (
    <>
      <section className="card">
        <div className="uploader">
          <input id="file" type="file" accept="image/*,video/*" onChange={(e) => pick(e.target.files?.[0] ?? null)} />
          <label
            htmlFor="file"
            className={`drop${dragOver ? ' over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            {file ? <strong>{file.name}</strong> : <span>{t('drop_prompt')}</span>}
          </label>
          <button className="primary" disabled={!file || loading} onClick={analyze}>
            {loading ? t('analyzing') : t('analyze')}
          </button>
        </div>
        {preview && !result && <img className="frame" src={preview} alt="preview" />}
        {error && <div className="error">{error}</div>}
      </section>

      {result && (
        <section className="card">
          {result.annotated && <img className="frame" src={`data:image/jpeg;base64,${result.annotated}`} alt="result" />}
          <h2>
            {t('plates_read', { n: reads.length })}
            {result.type === 'video' && result.frames ? t('frames_scanned', { frames: result.frames }) : ''}
          </h2>
          {reads.length > 0 ? (
            <table>
              <thead><tr><th>{t('th_plate')}</th><th>{t('th_region')}</th><th>{t('th_confidence')}</th></tr></thead>
              <tbody>
                {reads.map((p, i) => (
                  <tr key={i}><td className="plate">{p.plate_text}</td><td>{p.region}</td><td>{Math.round(p.confidence * 100)}%</td></tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">{t('no_read')}</p>
          )}
          <div className="actions">
            {reads.length > 0 && (
              <button className="ghost" onClick={copyCsv}>{copied ? t('copied') : t('copy_csv')}</button>
            )}
            <a className="csv" href={`${base}/api/results.csv`} target="_blank" rel="noreferrer">{t('download_csv')}</a>
          </div>
        </section>
      )}
    </>
  )
}
