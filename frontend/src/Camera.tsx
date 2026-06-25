import { useEffect, useRef, useState } from 'react'

/**
 * Live-camera AR overlay (best-effort). The graded classical detector runs on the BACKEND, so
 * we throttle frames to it (~2x/s); between detections each box GLIDES to its new target via
 * lerp and the recognized text LOCKS through multi-frame majority voting, giving an AR feel
 * without client-side CV. 2D only (no WebXR). iOS Safari needs HTTPS + camera permission.
 *
 * Boxes LINGER 3s after a plate leaves frame (time-based, independent of detection cadence)
 * and fade out over the last 1.5s so the last reading stays readable.
 */
const LINGER_MS = 3000 // keep a box this long after it was last detected
const FADE_MS = 1500 // start fading this long before removal

type Track = {
  x: number; y: number; w: number; h: number          // drawn (smoothed) box, in video px
  tx: number; ty: number; tw: number; th: number       // target box from the latest detection
  text: string | null
  votes: Record<string, number>
  lastSeen: number                                     // Date.now() of the last matched detection
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
  const [starting, setStarting] = useState(false)

  function attach(stream: MediaStream) {
    const v = videoRef.current!
    v.srcObject = stream
    return v.play()
  }

  // Try a sequence of constraint sets so the camera opens on ANY device. On a laptop the
  // 'environment' hint is soft (ideal), so it still grabs the front cam; if a camera is busy
  // (NotReadableError — common on Windows when Zoom/Teams/another tab holds it) we fall back
  // to a plain request and then to each enumerated device, which can grab a different free cam.
  async function start() {
    setErr(null)
    if (!window.isSecureContext) {
      setErr('Camera needs a secure (HTTPS) page. Open the github.io URL, not an http:// address or IP.')
      return
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setErr('This browser blocks camera access here. Use Safari (iOS) or Chrome/Edge (desktop) on the HTTPS site.')
      return
    }
    setStarting(true)
    stopStream() // release anything we might already hold

    const attempts: MediaStreamConstraints[] = [
      { video: { facingMode: { ideal: 'environment' } }, audio: false },
      { video: true, audio: false },
    ]
    let lastErr: Error | null = null
    for (const constraints of attempts) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints)
        await attach(stream)
        setOn(true); setStarting(false); return
      } catch (e) {
        lastErr = e as Error
      }
    }

    // Last resort: try every enumerated camera by id — one may be free even if the default is busy.
    try {
      const cams = (await navigator.mediaDevices.enumerateDevices())
        .filter((d) => d.kind === 'videoinput' && d.deviceId)
      for (const cam of cams) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: cam.deviceId } }, audio: false,
          })
          await attach(stream)
          setOn(true); setStarting(false); return
        } catch (e) { lastErr = e as Error }
      }
    } catch { /* enumerateDevices can throw on some browsers — fall through to the message */ }

    setStarting(false)
    const n = lastErr?.name
    if (n === 'NotReadableError' || n === 'AbortError')
      setErr('Camera is busy or blocked. Close other apps/tabs using it (Zoom, Teams, Windows Camera, Photo Booth), then check your OS camera privacy setting and retry. On Windows: Settings → Privacy & security → Camera → on.')
    else if (n === 'NotAllowedError' || n === 'SecurityError')
      setErr('Camera permission was denied. iPhone: tap "aA" in Safari\'s address bar → Website Settings → Camera → Allow, then reload. Desktop: click the camera icon in the address bar → Allow.')
    else if (n === 'NotFoundError' || n === 'OverconstrainedError')
      setErr('No camera was found on this device. Connect one (or use the Upload tab) and retry.')
    else
      setErr(`Camera error: ${n || lastErr?.message || 'unknown'}. Needs HTTPS + an available camera.`)
  }

  function stopStream() {
    const v = videoRef.current
    ;(v?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop())
    if (v) v.srcObject = null
  }

  function stop() {
    stopStream()
    tracksRef.current = []
    setOn(false)
  }

  function ingest(dets: Det[]) {
    const now = Date.now()
    const tracks = tracksRef.current
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
        Object.assign(best, { tx: bx, ty: by, tw: bw, th: bh, lastSeen: now })
        if (d.text) {
          best.votes[d.text] = (best.votes[d.text] || 0) + 1
          best.text = Object.entries(best.votes).sort((a, b) => b[1] - a[1])[0][0]
        }
      } else {
        tracks.push({ x: bx, y: by, w: bw, h: bh, tx: bx, ty: by, tw: bw, th: bh,
                      text: d.text, votes: d.text ? { [d.text]: 1 } : {}, lastSeen: now })
      }
    }
    // drop tracks not seen for LINGER_MS (time-based, so cadence-independent)
    tracksRef.current = tracks.filter((t) => now - t.lastSeen < LINGER_MS)
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
        const now = Date.now()
        for (const t of tracksRef.current) {
          const since = now - t.lastSeen
          if (since > LINGER_MS) continue
          // fade out over the last FADE_MS before removal so the final reading stays visible
          const alpha = since <= LINGER_MS - FADE_MS ? 1 : Math.max(0, (LINGER_MS - since) / FADE_MS)
          t.x += (t.tx - t.x) * 0.35; t.y += (t.ty - t.y) * 0.35
          t.w += (t.tw - t.w) * 0.35; t.h += (t.th - t.h) * 0.35
          const x = t.x * sx, y = t.y * sy, w = t.w * sx, h = t.h * sy
          ctx.globalAlpha = alpha
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
          ctx.globalAlpha = 1
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
      <button className="primary" onClick={on ? stop : start} disabled={starting}>
        {starting ? 'Opening camera…' : on ? 'Stop' : 'Start camera'}
      </button>
      <p className="muted">
        Point the rear camera at a Macedonian plate. Detection runs on the backend ~2×/s; the label glides
        and locks via multi-frame voting, and lingers ~3s after the plate leaves frame. Works best on clear, close plates.
      </p>
    </section>
  )
}
