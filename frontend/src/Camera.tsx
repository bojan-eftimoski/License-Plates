import { useEffect, useRef, useState } from 'react'

/**
 * Live-camera AR overlay (best-effort). The graded classical detector runs on the BACKEND, so
 * we throttle frames to it (~2x/s); between detections each box GLIDES to its new target via
 * lerp and the recognized text LOCKS through multi-frame majority voting, giving an AR feel
 * without client-side CV. 2D only (no WebXR). iOS Safari needs HTTPS + camera permission.
 */
type Track = {
  x: number; y: number; w: number; h: number          // drawn (smoothed) box, in video px
  tx: number; ty: number; tw: number; th: number       // target box from the latest detection
  text: string | null
  votes: Record<string, number>
  age: number                                          // detections since last seen
}

type Det = { bbox: number[]; text: string | null; conf: number }

export default function Camera({ base }: { base: string }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tracksRef = useRef<Track[]>([])
  const busyRef = useRef(false)
  const rafRef = useRef(0)
  const [on, setOn] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  async function start() {
    setErr(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setErr('Camera not available — open the HTTPS site (the github.io URL), not an http:// address. iOS blocks the camera on insecure pages.')
      return
    }
    try {
      let stream: MediaStream
      try {
        // soft 'ideal' rear-camera so it doesn't OverconstrainedError on odd devices
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false })
      } catch (e) {
        const n = (e as Error).name
        if (n === 'OverconstrainedError' || n === 'NotFoundError' || n === 'TypeError')
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        else throw e
      }
      const v = videoRef.current!
      v.srcObject = stream
      await v.play()
      setOn(true)
    } catch (e) {
      const n = (e as Error).name
      if (n === 'NotAllowedError')
        setErr('Camera permission is blocked. On iPhone: tap “aA” in the Safari address bar → Website Settings → Camera → Allow, then reload. Also check Settings → Safari → Camera = “Ask”/“Allow” (not Deny).')
      else
        setErr(`Camera error: ${n || (e as Error).message}. Needs HTTPS + camera permission.`)
    }
  }

  function stop() {
    const v = videoRef.current
    ;(v?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop())
    if (v) v.srcObject = null
    tracksRef.current = []
    setOn(false)
  }

  function ingest(dets: Det[]) {
    const tracks = tracksRef.current
    tracks.forEach((t) => (t.age += 1))
    for (const d of dets) {
      const [bx, by, bw, bh] = d.bbox
      const cx = bx + bw / 2, cy = by + bh / 2
      let best: Track | null = null
      let bestDist = Infinity
      for (const t of tracks) {
        const dist = Math.hypot(cx - (t.tx + t.tw / 2), cy - (t.ty + t.th / 2))
        if (dist < Math.max(bw, t.tw) && dist < bestDist) { best = t; bestDist = dist }
      }
      if (best) {
        Object.assign(best, { tx: bx, ty: by, tw: bw, th: bh, age: 0 })
        if (d.text) {
          best.votes[d.text] = (best.votes[d.text] || 0) + 1
          best.text = Object.entries(best.votes).sort((a, b) => b[1] - a[1])[0][0]
        }
      } else {
        tracks.push({ x: bx, y: by, w: bw, h: bh, tx: bx, ty: by, tw: bw, th: bh,
                      text: d.text, votes: d.text ? { [d.text]: 1 } : {}, age: 0 })
      }
    }
    tracksRef.current = tracks.filter((t) => t.age < 6)
  }

  // throttled backend detection
  useEffect(() => {
    if (!on) return
    const id = window.setInterval(async () => {
      const v = videoRef.current
      if (!v || !v.videoWidth || busyRef.current) return
      busyRef.current = true
      try {
        const scale = 640 / v.videoWidth
        const off = document.createElement('canvas')
        off.width = Math.round(v.videoWidth * scale)
        off.height = Math.round(v.videoHeight * scale)
        off.getContext('2d')!.drawImage(v, 0, 0, off.width, off.height)
        const blob = await new Promise<Blob>((res) => off.toBlob((b) => res(b!), 'image/jpeg', 0.7))
        const fd = new FormData()
        fd.append('file', new File([blob], 'frame.jpg', { type: 'image/jpeg' }))
        const j = await (await fetch(`${base}/api/analyze`, { method: 'POST', body: fd })).json()
        ingest((j.plates || []).map((p: { bbox: number[]; plate_text: string | null; confidence: number }) =>
          ({ bbox: p.bbox.map((n) => n / scale), text: p.plate_text, conf: p.confidence })))
      } catch { /* transient network/frame error — ignore */ }
      finally { busyRef.current = false }
    }, 500)
    return () => clearInterval(id)
  }, [on, base])

  // smooth render loop
  useEffect(() => {
    if (!on) return
    const draw = () => {
      const v = videoRef.current, c = canvasRef.current
      if (v && c && v.videoWidth) {
        c.width = v.clientWidth
        c.height = v.clientHeight
        const sx = c.width / v.videoWidth, sy = c.height / v.videoHeight
        const ctx = c.getContext('2d')!
        ctx.clearRect(0, 0, c.width, c.height)
        for (const t of tracksRef.current) {
          t.x += (t.tx - t.x) * 0.35; t.y += (t.ty - t.y) * 0.35
          t.w += (t.tw - t.w) * 0.35; t.h += (t.th - t.h) * 0.35
          const x = t.x * sx, y = t.y * sy, w = t.w * sx, h = t.h * sy
          ctx.lineWidth = 3
          ctx.strokeStyle = t.text ? '#22c55e' : '#ef4444'
          ctx.strokeRect(x, y, w, h)
          if (t.text) {
            ctx.font = 'bold 20px ui-monospace, monospace'
            const tw = ctx.measureText(t.text).width + 12
            ctx.fillStyle = '#22c55e'
            ctx.fillRect(x, y - 26, tw, 24)
            ctx.fillStyle = '#06240f'
            ctx.fillText(t.text, x + 6, y - 8)
          }
        }
      }
      rafRef.current = requestAnimationFrame(draw)
    }
    rafRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafRef.current)
  }, [on])

  useEffect(() => () => stop(), [])  // cleanup on unmount

  return (
    <section className="card camera">
      {err && <div className="error">{err}</div>}
      <div className="stage">
        <video ref={videoRef} playsInline muted autoPlay />
        <canvas ref={canvasRef} />
      </div>
      <button className="primary" onClick={on ? stop : start}>{on ? 'Stop' : 'Start camera'}</button>
      <p className="muted">
        Point the rear camera at a Macedonian plate. Detection runs on the backend ~2×/s; the label glides
        and locks via multi-frame voting. Works best on clear, close plates.
      </p>
    </section>
  )
}
