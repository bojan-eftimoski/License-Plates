import { useEffect, useRef, useState } from 'react'
import type { T } from './i18n'

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
const READ_COLOR = '#5f7d3b' // olive — confident read (harmonised with the editorial palette)
const DETECT_COLOR = '#b8432f' // brick — detected but unread

type Track = {
  x: number; y: number; w: number; h: number          // drawn (smoothed) box, in video px
  tx: number; ty: number; tw: number; th: number       // target box from the latest detection
  text: string | null
  votes: Record<string, number>
  lastSeen: number                                     // Date.now() of the last matched detection
}

type Det = { bbox: number[]; text: string | null; conf: number }

export default function Camera({ base, t }: { base: string; t: T }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const tracksRef = useRef<Track[]>([])
  const busyRef = useRef(false)
  const rafRef = useRef(0)
  const [on, setOn] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [fs, setFs] = useState(false)

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
      setErr(t('cam_insecure'))
      return
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setErr(t('cam_unsupported'))
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
      setErr(t('cam_busy'))
    else if (n === 'NotAllowedError' || n === 'SecurityError')
      setErr(t('cam_denied'))
    else if (n === 'NotFoundError' || n === 'OverconstrainedError')
      setErr(t('cam_none'))
    else
      setErr(t('cam_generic', { n: n || lastErr?.message || 'unknown' }))
  }

  function stopStream() {
    const v = videoRef.current
    ;(v?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop())
    if (v) v.srcObject = null
  }

  function stop() {
    exitFs()
    stopStream()
    tracksRef.current = []
    setOn(false)
  }

  // Fullscreen "AR" mode. iOS Safari has no element Fullscreen API, so the CSS-fixed `.fs`
  // overlay is the real mechanism (works everywhere, keeps the canvas overlay on top of the
  // video). Where the native API exists (desktop, Android) we ALSO request it to hide the
  // browser chrome for a more immersive feel — if it rejects (iOS) the CSS overlay still wins.
  function enterFs() {
    setFs(true)
    const el = stageRef.current as (HTMLElement & { webkitRequestFullscreen?: () => void }) | null
    if (!el) return
    try {
      if (el.requestFullscreen) el.requestFullscreen().catch(() => {})
      else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen()
    } catch { /* ignore — the CSS .fs overlay already provides fullscreen */ }
  }
  function exitFs() {
    setFs(false)
    const d = document as Document & { webkitFullscreenElement?: Element; webkitExitFullscreen?: () => void }
    if (d.fullscreenElement) d.exitFullscreen?.()
    else if (d.webkitFullscreenElement) d.webkitExitFullscreen?.()
  }

  // Keep our `fs` flag in sync when the user leaves native fullscreen via Esc / system gesture.
  useEffect(() => {
    const d = document as Document & { webkitFullscreenElement?: Element }
    const onChange = () => { if (!d.fullscreenElement && !d.webkitFullscreenElement) setFs(false) }
    document.addEventListener('fullscreenchange', onChange)
    document.addEventListener('webkitfullscreenchange', onChange)
    return () => {
      document.removeEventListener('fullscreenchange', onChange)
      document.removeEventListener('webkitfullscreenchange', onChange)
    }
  }, [])

  function ingest(dets: Det[]) {
    const now = Date.now()
    const tracks = tracksRef.current
    for (const d of dets) {
      const [bx, by, bw, bh] = d.bbox
      const cx = bx + bw / 2, cy = by + bh / 2
      let best: Track | null = null
      let bestDist = Infinity
      for (const trk of tracks) {
        const dist = Math.hypot(cx - (trk.tx + trk.tw / 2), cy - (trk.ty + trk.th / 2))
        if (dist < Math.max(bw, trk.tw) && dist < bestDist) { best = trk; bestDist = dist }
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
    tracksRef.current = tracks.filter((trk) => now - trk.lastSeen < LINGER_MS)
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
        // size & position the canvas exactly over the video's displayed box, so AR boxes stay
        // aligned even when the video is letterboxed/centered (fullscreen) rather than filling.
        const vw = v.clientWidth, vh = v.clientHeight
        c.width = vw; c.height = vh
        c.style.width = vw + 'px'; c.style.height = vh + 'px'
        c.style.left = v.offsetLeft + 'px'; c.style.top = v.offsetTop + 'px'
        const sx = vw / v.videoWidth, sy = vh / v.videoHeight
        const ctx = c.getContext('2d')!
        ctx.clearRect(0, 0, c.width, c.height)
        const now = Date.now()
        for (const trk of tracksRef.current) {
          const since = now - trk.lastSeen
          if (since > LINGER_MS) continue
          // fade out over the last FADE_MS before removal so the final reading stays visible
          const alpha = since <= LINGER_MS - FADE_MS ? 1 : Math.max(0, (LINGER_MS - since) / FADE_MS)
          trk.x += (trk.tx - trk.x) * 0.35; trk.y += (trk.ty - trk.y) * 0.35
          trk.w += (trk.tw - trk.w) * 0.35; trk.h += (trk.th - trk.h) * 0.35
          const x = trk.x * sx, y = trk.y * sy, w = trk.w * sx, h = trk.h * sy
          ctx.globalAlpha = alpha
          ctx.lineWidth = 3
          ctx.strokeStyle = trk.text ? READ_COLOR : DETECT_COLOR
          ctx.strokeRect(x, y, w, h)
          if (trk.text) {
            ctx.font = 'bold 20px ui-monospace, monospace'
            const tw = ctx.measureText(trk.text).width + 12
            ctx.fillStyle = READ_COLOR
            ctx.fillRect(x, y - 26, tw, 24)
            ctx.fillStyle = '#fbf7ed'
            ctx.fillText(trk.text, x + 6, y - 8)
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
      <div className={`stage${fs ? ' fs' : ''}`} ref={stageRef}>
        <video ref={videoRef} playsInline muted autoPlay />
        <canvas ref={canvasRef} />
        {fs && (
          <button className="fs-exit" onClick={exitFs} title={t('cam_exit')} aria-label={t('cam_exit')}>✕</button>
        )}
      </div>
      <div className="cam-controls">
        <button className="primary" onClick={on ? stop : start} disabled={starting}>
          {starting ? t('cam_opening') : on ? t('cam_stop') : t('cam_start')}
        </button>
        {on && !fs && (
          <button className="ghost" onClick={enterFs}>{t('cam_fullscreen')}</button>
        )}
      </div>
      <p className="muted">{t('cam_hint')}</p>
    </section>
  )
}
