import { useRef, useState, useEffect, useCallback } from 'react'
import { exportToExcel } from '../utils/export'

const MM_PX = 0.0064
const FRAME_INTERVAL_MS = 500   // analyze every 500 ms

export default function CameraAnalyzer() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)       // hidden — for capture
  const overlayRef = useRef(null)      // drawn on top of video
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const frameNumRef = useRef(0)

  const [cameras, setCameras] = useState([])
  const [selectedCam, setSelectedCam] = useState('')
  const [running, setRunning] = useState(false)
  const [liveMetrics, setLiveMetrics] = useState(null)
  const [frameLog, setFrameLog] = useState([])

  // Load camera list
  useEffect(() => {
    const load = async () => {
      try {
        // Request permission first so labels appear
        await navigator.mediaDevices.getUserMedia({ video: true }).then(s => s.getTracks().forEach(t => t.stop()))
        const devices = await navigator.mediaDevices.enumerateDevices()
        const cams = devices.filter(d => d.kind === 'videoinput')
        setCameras(cams)
        if (cams.length) setSelectedCam(cams[0].deviceId)
      } catch {
        setCameras([])
      }
    }
    load()
    return () => stopCamera()
  }, [])

  const startCamera = async () => {
    try {
      const constraints = {
        video: {
          deviceId: selectedCam ? { exact: selectedCam } : undefined,
          width: { ideal: 1280 }, height: { ideal: 720 },
        }
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      videoRef.current.srcObject = stream
      videoRef.current.style.display = 'block'
      document.getElementById('cam-placeholder').style.display = 'none'
      setRunning(true)
      timerRef.current = setInterval(processFrame, FRAME_INTERVAL_MS)
    } catch (e) {
      alert('Camera access denied or unavailable.\nPlease allow camera permission in your browser and try again.')
    }
  }

  const stopCamera = () => {
    clearInterval(timerRef.current)
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (videoRef.current) { videoRef.current.srcObject = null; videoRef.current.style.display = 'none' }
    const ph = document.getElementById('cam-placeholder')
    if (ph) ph.style.display = 'flex'
    setRunning(false)
  }

  const processFrame = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const overlay = overlayRef.current
    if (!video || !canvas || !overlay || video.readyState < 2) return

    const W = video.videoWidth, H = video.videoHeight
    canvas.width = W; canvas.height = H
    overlay.width = W; overlay.height = H

    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, W, H)
    const imgData = ctx.getImageData(0, 0, W, H)

    // Greyscale
    const gray = new Uint8Array(W * H)
    for (let i = 0; i < imgData.data.length; i += 4)
      gray[i / 4] = Math.round(0.299 * imgData.data[i] + 0.587 * imgData.data[i + 1] + 0.114 * imgData.data[i + 2])

    // Shannon entropy
    const hist = new Array(256).fill(0)
    for (const v of gray) hist[v]++
    const n = gray.length
    let Hval = 0
    for (const c of hist) { if (c > 0) { const p = c / n; Hval -= p * Math.log(p) } }
    const Hnorm = Hval / Math.log(256)

    // Bubble detection (lightweight threshold + circle estimation)
    const bubbles = detectBubbles(gray, W, H)

    // Derived
    const meanR_mm = bubbles.length ? bubbles.reduce((s, b) => s + b.r, 0) / bubbles.length * MM_PX : 0
    const maskSum = bubbles.reduce((s, b) => s + Math.PI * b.r * b.r, 0)
    const coverage = Math.min(99, maskSum / (W * H) * 100)

    // Draw overlay
    const octx = overlay.getContext('2d')
    octx.clearRect(0, 0, W, H)
    octx.strokeStyle = 'rgba(255,80,80,0.9)'; octx.lineWidth = 1.5
    bubbles.forEach(b => {
      octx.beginPath(); octx.arc(b.x, b.y, b.r, 0, Math.PI * 2); octx.stroke()
    })
    // HUD
    octx.fillStyle = 'rgba(0,0,0,0.55)'; octx.fillRect(4, 4, 280, 70)
    octx.fillStyle = '#58a6ff'; octx.font = '12px monospace'
    octx.fillText(`Frame ${frameNumRef.current}  |  Bubbles: ${bubbles.length}`, 10, 22)
    octx.fillText(`H = ${Hval.toFixed(3)}  H_norm = ${(Hnorm * 100).toFixed(1)}%`, 10, 40)
    octx.fillText(`Mean Ø = ${(meanR_mm * 2).toFixed(3)} mm  Cov = ${coverage.toFixed(1)}%`, 10, 58)

    const ts = new Date().toISOString().slice(11, 23)
    const row = {
      frame: ++frameNumRef.current,
      time: ts,
      bubbles: bubbles.length,
      H: +Hval.toFixed(4),
      Hnorm: +(Hnorm * 100).toFixed(2),
      meanDiam_mm: +(meanR_mm * 2).toFixed(4),
      coverage_pct: +coverage.toFixed(2),
    }

    setLiveMetrics(row)
    setFrameLog(prev => [row, ...prev].slice(0, 500))
  }, [])

  // CORRECT detection: real H2 bubbles are DARK quasi-circular blobs on grey electrode
  function detectBubbles(gray, W, H) {
    // 1. Gaussian blur (box approx)
    const k = 7, half = 3
    const tmp = new Uint8Array(W * H)
    const blr = new Uint8Array(W * H)
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let s=0,c=0; for(let d=-half;d<=half;d++){const xx=Math.max(0,Math.min(W-1,x+d));s+=gray[y*W+xx];c++} tmp[y*W+x]=Math.round(s/c)
      }
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let s=0,c=0; for(let d=-half;d<=half;d++){const yy=Math.max(0,Math.min(H-1,y+d));s+=tmp[yy*W+x];c++} blr[y*W+x]=Math.round(s/c)
      }

    // 2. Adaptive threshold (inverted) — dark pixels relative to local mean → 255
    const blockH = 61, bh2 = 30, C_thresh = 5
    const intg = new Float64Array((W+1)*(H+1))
    for(let y=0;y<H;y++) for(let x=0;x<W;x++)
      intg[(y+1)*(W+1)+(x+1)] = blr[y*W+x]+intg[y*(W+1)+(x+1)]+intg[(y+1)*(W+1)+x]-intg[y*(W+1)+x]
    const thresh = new Uint8Array(W*H)
    for(let y=0;y<H;y++) for(let x=0;x<W;x++) {
      const x1=Math.max(0,x-bh2),y1=Math.max(0,y-bh2),x2=Math.min(W-1,x+bh2),y2=Math.min(H-1,y+bh2)
      const area=(x2-x1+1)*(y2-y1+1)
      const sum=intg[(y2+1)*(W+1)+(x2+1)]-intg[y1*(W+1)+(x2+1)]-intg[(y2+1)*(W+1)+x1]+intg[y1*(W+1)+x1]
      thresh[y*W+x] = blr[y*W+x] < sum/area - C_thresh ? 255 : 0
    }

    // 3. Simple morphological open (erode then dilate, r=4)
    const er = new Uint8Array(W*H), cl = new Uint8Array(W*H)
    const mr = 4
    for(let y=0;y<H;y++) for(let x=0;x<W;x++){
      let mn=255; for(let dy=-mr;dy<=mr;dy++) for(let dx=-mr;dx<=mr;dx++){if(dx*dx+dy*dy>mr*mr)continue;const ny2=Math.max(0,Math.min(H-1,y+dy)),nx2=Math.max(0,Math.min(W-1,x+dx));mn=Math.min(mn,thresh[ny2*W+nx2])} er[y*W+x]=mn
    }
    for(let y=0;y<H;y++) for(let x=0;x<W;x++){
      let mx=0; for(let dy=-mr;dy<=mr;dy++) for(let dx=-mr;dx<=mr;dx++){if(dx*dx+dy*dy>mr*mr)continue;const ny2=Math.max(0,Math.min(H-1,y+dy)),nx2=Math.max(0,Math.min(W-1,x+dx));mx=Math.max(mx,er[ny2*W+nx2])} cl[y*W+x]=mx
    }

    // 4. Connected components → circles
    const labels = new Int32Array(W*H).fill(-1)
    const bubbles = []
    let lid = 0, queue = []
    for(let y=0;y<H;y++) for(let x=0;x<W;x++){
      if(cl[y*W+x]!==255||labels[y*W+x]!==-1) continue
      const pixels=[]; queue=[y*W+x]; labels[y*W+x]=lid
      let qi=0
      while(qi<queue.length){
        const idx=queue[qi++]; pixels.push(idx)
        const py=Math.floor(idx/W),px=idx%W
        for(const[ny2,nx2] of [[py-1,px],[py+1,px],[py,px-1],[py,px+1]]){
          if(ny2<0||ny2>=H||nx2<0||nx2>=W)continue
          const ni=ny2*W+nx2; if(cl[ni]===255&&labels[ni]===-1){labels[ni]=lid;queue.push(ni)}
        }
      }
      lid++
      const area=pixels.length; if(area<150||area>40000) continue
      let minX2=W,maxX2=0,minY2=H,maxY2=0,cx2=0,cy2=0
      for(const idx of pixels){const py=Math.floor(idx/W),px2=idx%W;minX2=Math.min(minX2,px2);maxX2=Math.max(maxX2,px2);minY2=Math.min(minY2,py);maxY2=Math.max(maxY2,py);cx2+=px2;cy2+=py}
      cx2/=pixels.length; cy2/=pixels.length
      const bw2=maxX2-minX2+1,bh3=maxY2-minY2+1,r=(bw2+bh3)/4
      if(r<8||r>100) continue
      const perim=2*Math.PI*r, circ=4*Math.PI*area/(perim*perim)
      if(circ<0.35) continue
      const asp=bw2>bh3?bw2/bh3:bh3/bw2; if(asp>2.2) continue
      bubbles.push({x:Math.round(cx2),y:Math.round(cy2),r:Math.round(r)})
      if(bubbles.length>=300) return bubbles
    }
    return bubbles
  }

  const exportCSV = () => {
    if (!frameLog.length) { alert('No frame data yet'); return }
    exportToExcel(frameLog, ['frame','time','bubbles','H','Hnorm','meanDiam_mm','coverage_pct'], 'camera_frame_log.xlsx')
  }

  const captureNow = () => { if (running) processFrame() }
  const clearLog = () => { setFrameLog([]); frameNumRef.current = 0 }

  return (
    <div className="panel">
      {/* Camera controls bar */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 16 }}>
        <select value={selectedCam} onChange={e => setSelectedCam(e.target.value)} style={{ maxWidth: 260 }}>
          {cameras.length
            ? cameras.map((c, i) => <option key={c.deviceId} value={c.deviceId}>{c.label || `Camera ${i + 1}`}</option>)
            : <option>No cameras found — allow permission</option>}
        </select>
        <button className={`btn ${running ? 'danger' : 'primary'}`} onClick={running ? stopCamera : startCamera}>
          {running ? '⏹ Stop camera' : '▶ Start camera'}
        </button>
        <button className="btn" onClick={captureNow} disabled={!running}>📸 Capture frame</button>
        <button className="btn success" onClick={exportCSV}>⬇ Export Excel</button>
        <button className="btn" onClick={clearLog}>🗑 Clear log</button>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
          <span className={`dot${running ? ' live' : ''}`}></span>
          <span style={{ color: running ? 'var(--green)' : 'var(--text2)' }}>{running ? 'Live' : 'Camera off'}</span>
        </div>
      </div>

      <div className="grid-2-equal">
        <div>
          {/* Video + overlay */}
          <div style={{ position: 'relative', background: '#000', borderRadius: 8, overflow: 'hidden', minHeight: 300 }}>
            <video ref={videoRef} autoPlay muted playsInline style={{ display: 'none', width: '100%', maxHeight: 380, objectFit: 'cover' }} />
            <canvas ref={overlayRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            <div id="cam-placeholder" style={{ height: 300, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, color: 'var(--text2)' }}>
              <div style={{ fontSize: 40 }}>📷</div>
              <span style={{ fontSize: 13 }}>Select camera and click Start</span>
            </div>
          </div>

          {/* Live metrics */}
          <div className="card section-gap">
            <div className="card-title">Live frame metrics</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
              {[
                { label: 'Frame #', val: liveMetrics?.frame ?? '—' },
                { label: 'Bubbles', val: liveMetrics?.bubbles ?? '—', cls: 'blue' },
                { label: 'Shannon H', val: liveMetrics?.H ?? '—', cls: 'blue' },
                { label: 'H_norm %', val: liveMetrics ? liveMetrics.Hnorm + '%' : '—', cls: 'blue' },
                { label: 'Mean Ø (mm)', val: liveMetrics?.meanDiam_mm ?? '—' },
                { label: 'Coverage %', val: liveMetrics?.coverage_pct ?? '—' },
              ].map(m => (
                <div key={m.label} className="metric">
                  <div className="metric-label">{m.label}</div>
                  <div className={`metric-value ${m.cls || ''}`} style={{ fontSize: 16 }}>{m.val}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Frame log */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
            Frame log <span style={{ color: 'var(--text2)', fontWeight: 400 }}>{frameLog.length} frames recorded</span>
          </div>
          <div className="log-scroll" style={{ flex: 1 }}>
            <table className="log-table">
              <thead><tr>
                <th>Frame</th><th>Time</th><th>Bubbles</th><th>H</th><th>H_norm%</th><th>Ø mm</th><th>Cov%</th>
              </tr></thead>
              <tbody>
                {frameLog.map(r => (
                  <tr key={r.frame}>
                    <td>{r.frame}</td><td>{r.time}</td><td>{r.bubbles}</td>
                    <td>{r.H}</td><td>{r.Hnorm}</td><td>{r.meanDiam_mm}</td><td>{r.coverage_pct}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ marginTop: 12, fontSize: 12, color: 'var(--text2)' }}>
            Frame captured every 500 ms · up to 500 rows in view · export saves all
          </div>
        </div>
      </div>
    </div>
  )
}
