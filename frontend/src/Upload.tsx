import { useState } from 'react'

type Plate = { plate_text: string | null; region: string | null; confidence: number; frame?: number }
type AnalyzeResult = { type: 'image' | 'video'; plates: Plate[]; annotated?: string | null; frames?: number }

export default function Upload({ base }: { base: string }) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalyzeResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  function pick(f: File | null) {
    setFile(f)
    setResult(null)
    setError(null)
    setPreview(f && f.type.startsWith('image') ? URL.createObjectURL(f) : null)
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
      setError(`Couldn't reach the backend at ${base}. Is it running / awake? (${(e as Error).message})`)
    } finally {
      setLoading(false)
    }
  }

  const reads = result?.plates.filter((p) => p.plate_text) ?? []

  return (
    <>
      <section className="card">
        <div className="uploader">
          <input id="file" type="file" accept="image/*,video/*" onChange={(e) => pick(e.target.files?.[0] ?? null)} />
          <label htmlFor="file" className="drop">
            {file ? <strong>{file.name}</strong> : <span>Choose an image or video…</span>}
          </label>
          <button className="primary" disabled={!file || loading} onClick={analyze}>
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>
        {preview && !result && <img className="frame" src={preview} alt="preview" />}
        {error && <div className="error">{error}</div>}
      </section>

      {result && (
        <section className="card">
          {result.annotated && <img className="frame" src={`data:image/jpeg;base64,${result.annotated}`} alt="result" />}
          <h2>
            {reads.length} plate{reads.length === 1 ? '' : 's'} read
            {result.type === 'video' && result.frames ? ` · ${result.frames} frames scanned` : ''}
          </h2>
          {reads.length > 0 ? (
            <table>
              <thead><tr><th>Plate</th><th>Region</th><th>Confidence</th></tr></thead>
              <tbody>
                {reads.map((p, i) => (
                  <tr key={i}><td className="plate">{p.plate_text}</td><td>{p.region}</td><td>{Math.round(p.confidence * 100)}%</td></tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">
              No plate read confidently. Red boxes mark detected plate regions the model couldn't read
              (too small / blurred / angled); green boxes are confident reads.
            </p>
          )}
          <a className="csv" href={`${base}/api/results.csv`} target="_blank" rel="noreferrer">⬇ Download results CSV</a>
        </section>
      )}
    </>
  )
}
