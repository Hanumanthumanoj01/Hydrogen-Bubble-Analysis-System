import { useRef, useState } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { exportToExcel } from '../utils/export'

// The recording is decoded by the backend with OpenCV rather than by the
// browser. That is what lets this tab open the laboratory's own AVI files and
// run the same detector the Image tab uses on every sampled frame. A browser
// cannot decode AVI at all, so the previous in-browser approach could neither
// open these files nor reach the Python detector.

export default function VideoAnalyzer() {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [shot, setShot] = useState(0)
  const [zipping, setZipping] = useState(false)
  const fileRef = useRef()

  const [params, setParams] = useState({
    current_density: 200,
    electrode_type: 'ans',
    cathode_area: 2.54,
    detector: 'primary',
    min_pct: 3,
    every: 5,
    max_frames: 60,
    camera_fps: 0,
  })

  const setP = (k, v) => setParams(p => ({ ...p, [k]: v }))

  const pickFile = (f) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setShot(0)
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('current_density', params.current_density)
    fd.append('electrode_type', params.electrode_type)
    fd.append('cathode_area', params.cathode_area)
    fd.append('detector', params.detector)
    fd.append('min_pct', params.min_pct)
    fd.append('every', params.every)
    fd.append('max_frames', params.max_frames)
    fd.append('camera_fps', params.camera_fps)
    try {
      const { data } = await axios.post('/analyze-video', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 0,            // a long recording legitimately takes minutes
      })
      setResult(data)
      setShot(0)
    } catch (e) {
      alert(e?.response?.data?.detail ||
            'Backend not reachable — start it with:\n\nuvicorn main:app --reload --port 8000')
    }
    setLoading(false)
  }

  // The browser is given shrunk JPEGs so the page stays usable; this asks the
  // backend for the full-resolution PNGs plus a CSV, as one zip.
  const downloadFrames = async () => {
    if (!file) return
    setZipping(true)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('current_density', params.current_density)
    fd.append('electrode_type', params.electrode_type)
    fd.append('cathode_area', params.cathode_area)
    fd.append('detector', params.detector)
    fd.append('min_pct', params.min_pct)
    fd.append('every', params.every)
    fd.append('max_frames', params.max_frames)
    fd.append('camera_fps', params.camera_fps)
    try {
      const resp = await axios.post('/analyze-video-frames', fd, {
        responseType: 'blob', timeout: 0,
      })
      const url = URL.createObjectURL(resp.data)
      const a = document.createElement('a')
      a.href = url
      a.download = (file.name || 'clip').replace(/\.[^.]+$/, '') + '_frames.zip'
      document.body.appendChild(a); a.click(); a.remove()
      URL.revokeObjectURL(url)
    } catch (e) {
      alert('Could not build the zip — is the backend running?')
    }
    setZipping(false)
  }

  const exportResults = () => {
    if (!result) return
    exportToExcel(
      result.per_frame,
      ['frame', 't_s', 'bubbles', 'H', 'H_norm', 'mean_diameter_mm', 'coverage_pct', 'H2_mL_per_min'],
      'video_analysis.xlsx',
      {
        sheet2: {
          name: 'Summary',
          data: [
            { metric: 'Source file', value: result.filename },
            { metric: 'Frames in file', value: result.frames_in_file },
            { metric: 'Frames analysed', value: result.frames_analyzed },
            { metric: 'Sampled every', value: result.sampled_every },
            { metric: 'Frame rate (fps)', value: result.fps },
            { metric: 'Frame rate source', value: result.fps_source },
            { metric: 'Detector', value: result.detector },
            { metric: 'Mean bubbles per frame', value: result.mean_bubbles },
            { metric: 'Fewest bubbles in a frame', value: result.min_bubbles },
            { metric: 'Most bubbles in a frame', value: result.max_bubbles },
            { metric: 'Mean Shannon H', value: result.mean_H },
            { metric: 'Mean H_norm (%)', value: result.mean_H_norm },
            { metric: 'Mean diameter (mm)', value: result.mean_diameter_mm },
            { metric: 'Mean coverage (%)', value: result.mean_coverage_pct },
            { metric: 'Mean H2 (mL/min)', value: result.mean_H2_mL_per_min },
            { metric: 'Current density (mA/cm²)', value: result.current_density },
            { metric: 'Cathode area (cm²)', value: result.cathode_area },
            { metric: 'Electrode type', value: result.electrode_type },
          ],
          cols: ['metric', 'value'],
        },
      })
  }

  const frames = result?.intervals || result?.previews || []

  return (
    <div className="panel">
      <div className="grid-2-equal">
        {/* Left column */}
        <div style={{ minWidth: 0 }}>
          <div className="card">
            <div className="card-title">Upload recording</div>
            <div
              className={`drop-zone${dragging ? ' dragover' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]) }}
              onClick={() => fileRef.current.click()}
            >
              <div style={{ fontSize: 32 }}>🎬</div>
              <p>Click or drag to upload<br />
                <span style={{ fontSize: 11 }}>AVI / MP4 / MOV / MKV — the laboratory's AVI files work as they are</span>
              </p>
            </div>
            <input ref={fileRef} type="file" accept="video/*,.avi,.mkv" style={{ display: 'none' }}
              onChange={e => pickFile(e.target.files[0])} />

            {file && (
              <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text2)' }}>
                <span style={{ fontSize: 14 }}>🎞</span>
                <span style={{ color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{file.name}</span>
                <span style={{ marginLeft: 'auto', whiteSpace: 'nowrap' }}>{(file.size / 1048576).toFixed(1)} MB</span>
              </div>
            )}

            <div style={{ marginTop: 14 }}>
              <div className="ctrl-section">
                <label className="ctrl-label">Current density <span>{params.current_density} mA/cm²</span></label>
                <input type="range" min="50" max="1000" step="10" value={params.current_density}
                  onChange={e => setP('current_density', +e.target.value)} />
              </div>
              <div className="ctrl-section">
                <label className="ctrl-label">Cathode area <span>{params.cathode_area} cm²</span></label>
                <input type="range" min="1" max="10" step="0.1" value={params.cathode_area}
                  onChange={e => setP('cathode_area', +e.target.value)} />
              </div>
              <div className="ctrl-section">
                <label className="ctrl-label">What to detect</label>
                <select value={params.detector}
                  onChange={e => {
                    const v = e.target.value
                    // Ring bubbles are large and sit among froth, so they need a
                    // higher floor than the plate close-ups. Moving the slider
                    // with the mode saves the user discovering that by hand.
                    setParams(p => ({ ...p, detector: v, min_pct: v === 'ring' ? 13 : 3 }))
                  }}>
                  <option value="primary">Bubbles on the plate — dark centre</option>
                  <option value="ring">Large detached bubbles — bright rim, froth present</option>
                  <option value="standard">Every bubble, froth included</option>
                </select>
                <p style={{ fontSize: 11, color: 'var(--text2)', marginTop: 4 }}>
                  {params.detector === 'ring'
                    ? 'For the "bigbubbles" recordings: transparent spheres in the electrolyte, seen as a bright ring rather than a black disc.'
                    : params.detector === 'primary'
                      ? 'For the nucleation-site recordings: each bubble has a near-black centre.'
                      : 'The original detector. Reports everything it can find, froth included.'}
                </p>
              </div>
              {params.detector !== 'standard' && (
                <div className="ctrl-section">
                  <label className="ctrl-label">
                    Smallest bubble to detect <span>{params.min_pct} % of frame height</span>
                  </label>
                  <input type="range" min="2" max="25" step="1" value={params.min_pct}
                    onChange={e => setP('min_pct', +e.target.value)} />
                </div>
              )}
              <div className="ctrl-section">
                <label className="ctrl-label">Electrode type</label>
                <select value={params.electrode_type} onChange={e => setP('electrode_type', e.target.value)}>
                  <option value="gde">Gas Diffusion Electrode (GDE)</option>
                  <option value="mesh">Expanded Nickel Mesh</option>
                  <option value="ans">Artificial Nucleation Sites (ANS)</option>
                </select>
              </div>
              <div className="ctrl-section">
                <label className="ctrl-label">Camera frame rate</label>
                <select value={params.camera_fps} onChange={e => setP('camera_fps', +e.target.value)}>
                  <option value={0}>As stored in the file</option>
                  <option value={100}>100 fps</option>
                  <option value={200}>200 fps</option>
                  <option value={500}>500 fps</option>
                  <option value={1000}>1000 fps</option>
                </select>
                <p style={{ fontSize: 11, color: 'var(--text2)', marginTop: 4 }}>
                  Used only for the time column. The laboratory files store 24 fps; the real rate is
                  printed in the top-left corner of the frame (e.g. "640 x 480 / 200 fps").
                </p>
              </div>
              <div className="ctrl-section">
                <label className="ctrl-label">Analyse every <span>{params.every} frames</span></label>
                <input type="range" min="1" max="25" step="1" value={params.every}
                  onChange={e => setP('every', +e.target.value)} />
              </div>
              <div className="ctrl-section">
                <label className="ctrl-label">Stop after <span>{params.max_frames} frames</span></label>
                <input type="range" min="10" max="300" step="10" value={params.max_frames}
                  onChange={e => setP('max_frames', +e.target.value)} />
                <p style={{ fontSize: 11, color: 'var(--text2)', marginTop: 4 }}>
                  {(() => {
                    // Measured per-frame cost of each detector on a 640 × 480 frame.
                    // The ring detector searches far more candidate circles, which is
                    // why it is an order of magnitude slower.
                    const per = { primary: 0.35, standard: 2.2, ring: 5.6 }[params.detector] || 1
                    const secs = Math.max(1, Math.round(params.max_frames * per))
                    return `About ${per} s per frame with this detector, so ${params.max_frames} frames takes roughly ${secs < 90 ? secs + ' seconds' : Math.round(secs / 60) + ' minutes'}.`
                  })()}
                </p>
              </div>
            </div>

            <button className="btn primary" style={{ marginTop: 14, width: '100%' }}
              disabled={!file || loading} onClick={analyze}>
              {loading ? 'Analyzing frame by frame…' : '▶ Analyze video'}
            </button>
            {loading && (
              <p style={{ fontSize: 11, color: 'var(--text2)', marginTop: 6, textAlign: 'center' }}>
                Decoding and detecting on the server. Leave this tab open.
              </p>
            )}
          </div>

          {result && (
            <div className="card section-gap">
              <div className="card-title">Averages over {result.frames_analyzed} frames</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                {[
                  { label: 'Mean bubbles / frame', val: result.mean_bubbles, cls: 'blue' },
                  { label: 'Fewest / most', val: `${result.min_bubbles} / ${result.max_bubbles}` },
                  { label: 'Frames in file', val: result.frames_in_file },
                  { label: 'Mean Shannon H', val: result.mean_H, cls: 'blue' },
                  { label: 'Mean H_norm %', val: result.mean_H_norm + '%', cls: 'blue' },
                  { label: 'Frame rate', val: (result.fps || '—') + ' fps' + (result.fps_source === 'file container' ? ' (file)' : '') },
                  { label: 'Mean diameter', val: result.mean_diameter_mm + ' mm' },
                  { label: 'Mean coverage', val: result.mean_coverage_pct + '%' },
                  { label: 'Mean H₂', val: result.mean_H2_mL_per_min + ' mL/min', cls: 'green' },
                ].map(m => (
                  <div key={m.label} className="metric">
                    <div className="metric-label">{m.label}</div>
                    <div className={`metric-value ${m.cls || ''}`} style={{ fontSize: 15 }}>{m.val}</div>
                  </div>
                ))}
              </div>
              <button className="btn success" style={{ marginTop: 12, width: '100%' }} onClick={exportResults}>
                ⬇ Export to Excel (per frame + summary)
              </button>
            </div>
          )}
        </div>

        {/* Right column */}
        <div style={{ minWidth: 0 }}>
          <div className="card">
            {/* Every sampled interval, not only three. The strip steps through
                them; the slider scrubs; the arrows move one interval at a time. */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              <button className="btn" style={{ fontSize: 12 }} disabled={!frames.length || shot <= 0}
                onClick={() => setShot(s => Math.max(0, s - 1))}>◀</button>
              <span style={{ fontSize: 12, color: 'var(--text)', minWidth: 150 }}>
                {frames.length
                  ? <>Interval {shot + 1} / {frames.length} · frame {frames[shot]?.frame}
                      {frames[shot]?.t_s != null && <> · {frames[shot].t_s}s</>}</>
                  : 'No frames yet'}
              </span>
              <button className="btn" style={{ fontSize: 12 }} disabled={!frames.length || shot >= frames.length - 1}
                onClick={() => setShot(s => Math.min(frames.length - 1, s + 1))}>▶</button>
              {frames[shot]?.bubbles != null && (
                <span className="badge" style={{ fontSize: 11 }}>
                  {frames[shot].bubbles} bubbles · H {frames[shot].H}
                </span>
              )}
              {result && (
                <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 220 }}>
                  {result.filename}
                </span>
              )}
            </div>

            {frames.length > 1 && (
              <input type="range" min="0" max={frames.length - 1} step="1" value={shot}
                onChange={e => setShot(+e.target.value)} style={{ marginBottom: 10 }} />
            )}

            <div style={{ minHeight: 300, background: '#000', borderRadius: 6, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {frames[shot]
                ? <img src={'data:image/png;base64,' + frames[shot].image} alt="Detections on this frame"
                    style={{ width: '100%', maxHeight: 420, objectFit: 'contain' }} />
                : <div style={{ color: 'var(--text2)', textAlign: 'center' }}>
                    <div style={{ fontSize: 40 }}>🎬</div>
                    <p style={{ marginTop: 8, fontSize: 13 }}>
                      {file ? 'Press Analyze video' : 'Upload a recording to begin'}
                    </p>
                  </div>}
            </div>

            {frames.length > 1 && (
              <div style={{ display: 'flex', gap: 4, overflowX: 'auto', marginTop: 10, paddingBottom: 4 }}>
                {frames.map((p, i) => (
                  <img key={p.frame} src={'data:image/png;base64,' + p.image}
                    alt={'frame ' + p.frame} title={'frame ' + p.frame}
                    onClick={() => setShot(i)}
                    style={{
                      height: 54, cursor: 'pointer', borderRadius: 3, flexShrink: 0,
                      border: shot === i ? '2px solid var(--blue)' : '2px solid transparent',
                      opacity: shot === i ? 1 : 0.65,
                    }} />
                ))}
              </div>
            )}

            {result && (
              <>
                <p style={{ fontSize: 11, color: 'var(--text2)', marginTop: 8 }}>
                  Every analysed interval, with the detections drawn on it. Click a thumbnail
                  or drag the slider. Counts changing between intervals is the physics —
                  a site reporting nothing is a site whose bubble has detached.
                </p>
                <button className="btn" style={{ marginTop: 8, width: '100%' }}
                  disabled={zipping} onClick={downloadFrames}>
                  {zipping ? 'Preparing…' : '⬇ Download all interval screenshots (full size, + CSV)'}
                </button>
              </>
            )}
          </div>

          {result && (
            <div className="card section-gap">
              <div className="card-title">Bubble count and entropy over the recording</div>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={result.per_frame}>
                  <CartesianGrid stroke="#30363d" strokeDasharray="3 3" />
                  <XAxis dataKey="frame" tick={{ fontSize: 10, fill: '#8b949e' }}
                    label={{ value: 'Frame', position: 'insideBottom', offset: -2, fill: '#8b949e', fontSize: 10 }} />
                  <YAxis yAxisId="l" tick={{ fontSize: 10, fill: '#8b949e' }} />
                  <YAxis yAxisId="r" orientation="right" tick={{ fontSize: 10, fill: '#8b949e' }} domain={['auto', 'auto']} />
                  <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', fontSize: 12 }} />
                  <Line yAxisId="l" type="monotone" dataKey="bubbles" stroke="#58a6ff" dot={false} strokeWidth={2} />
                  <Line yAxisId="r" type="monotone" dataKey="H_norm" stroke="#3fb950" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
              <p style={{ fontSize: 11, color: 'var(--text2)' }}>
                Blue: bubbles detected (left axis). Green: H_norm % (right axis).
              </p>
            </div>
          )}

          {result && (
            <div className="card section-gap">
              <div className="card-title">Per-frame results</div>
              <div style={{ maxHeight: 260, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ color: 'var(--text2)', textAlign: 'right' }}>
                      <th style={{ textAlign: 'left', padding: '4px 6px' }}>Frame</th>
                      <th style={{ padding: '4px 6px' }}>t (s)</th>
                      <th style={{ padding: '4px 6px' }}>Bubbles</th>
                      <th style={{ padding: '4px 6px' }}>H</th>
                      <th style={{ padding: '4px 6px' }}>H_norm %</th>
                      <th style={{ padding: '4px 6px' }}>Mean d (mm)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.per_frame.map(r => (
                      <tr key={r.frame} style={{ borderTop: '1px solid #30363d', textAlign: 'right' }}>
                        <td style={{ textAlign: 'left', padding: '3px 6px' }}>{r.frame}</td>
                        <td style={{ padding: '3px 6px' }}>{r.t_s ?? '—'}</td>
                        <td style={{ padding: '3px 6px', color: 'var(--blue)' }}>{r.bubbles}</td>
                        <td style={{ padding: '3px 6px' }}>{r.H}</td>
                        <td style={{ padding: '3px 6px' }}>{r.H_norm}</td>
                        <td style={{ padding: '3px 6px' }}>{r.mean_diameter_mm}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
