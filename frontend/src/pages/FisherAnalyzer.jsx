import { useRef, useState } from 'react'
import axios from 'axios'

// This tab detects nothing. That is the point of it.
//
// The Image Analyzer answers "where are the bubbles and how many". This one
// answers "what does the brightness field itself look like" — the histogram,
// its Shannon entropy, its Fisher information, and the spatial Fisher maps.
// Because no detector is involved anywhere, the numbers here are an
// independent check on the detector rather than a restatement of it.

export default function FisherAnalyzer() {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [view, setView] = useState('combined')
  const fileRef = useRef()

  const [params, setParams] = useState({
    n_peaks: 20,
    hist_smooth: 2.0,
    mask_timestamp: true,
  })
  const setP = (k, v) => setParams(p => ({ ...p, [k]: v }))

  const pickFile = (f) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setPreview(URL.createObjectURL(f))
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('n_peaks', params.n_peaks)
    fd.append('hist_smooth', params.hist_smooth)
    fd.append('mask_timestamp', params.mask_timestamp)
    try {
      const { data } = await axios.post('/analyze-fisher', fd, {
        headers: { 'Content-Type': 'multipart/form-data' }, timeout: 0,
      })
      setResult(data)
      setView('combined')
    } catch (e) {
      alert(e?.response?.data?.detail ||
        'Backend not reachable — start it with:\n\nuvicorn main:app --reload --port 8000')
    }
    setLoading(false)
  }

  // Excel is built on the backend with openpyxl, three sheets: Histogram,
  // Summary, Peaks. Doing it there rather than in the browser means the file
  // does not depend on what the browser-side export helper supports.
  const [exporting, setExporting] = useState(false)
  const exportResults = async () => {
    if (!file) return
    setExporting(true)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('n_peaks', params.n_peaks)
    fd.append('hist_smooth', params.hist_smooth)
    fd.append('mask_timestamp', params.mask_timestamp)
    try {
      const resp = await axios.post('/analyze-fisher-xlsx', fd, { responseType: 'blob', timeout: 0 })
      const url = URL.createObjectURL(resp.data)
      const a = document.createElement('a')
      a.href = url
      a.download = (file.name || 'picture').replace(/\.[^.]+$/, '') + '_fisher.xlsx'
      document.body.appendChild(a); a.click(); a.remove()
      URL.revokeObjectURL(url)
    } catch (e) {
      alert('Excel export failed. If the backend says openpyxl is missing:\n\npip3 install openpyxl')
    }
    setExporting(false)
  }

  const maps = result ? [
    ['combined', '🔥 F = Fx + Fy', result.fisher_combined_image],
    ['x', '↔ Fx  (across)', result.fisher_x_image],
    ['y', '↕ Fy  (down)', result.fisher_y_image],
    ['peaks', '📍 Peaks', result.peaks_image],
    ['source', '🖼 Original', null],
  ] : []

  const shown = result && (view === 'source'
    ? preview
    : 'data:image/png;base64,' + (maps.find(m => m[0] === view) || [])[2])

  // One line that says what the x/y ratio means, so the number is not just a number.
  const ratioNote = (r) => {
    if (r == null) return ''
    if (r > 1.25) return 'brightness runs mostly across the frame — streaks or machining marks dominate'
    if (r < 0.8) return 'brightness runs mostly down the frame — horizontal banding dominates'
    return 'close to 1 — structures are round or evenly oriented, as a bubble field should be'
  }

  const edgeNote = (e) => {
    if (e == null) return ''
    if (e < 1.5) return 'low — the map is spread evenly, which is what noise looks like'
    if (e < 3) return 'moderate — the map has found boundaries'
    return 'strong — the information sits clearly on the boundaries'
  }

  return (
    <div className="panel">
      <div className="grid-2-equal">
        {/* ── left ───────────────────────────────────────────────── */}
        <div style={{ minWidth: 0 }}>
          <div className="card">
            <div className="card-title">Upload a picture</div>
            <div
              className={`drop-zone${dragging ? ' dragover' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]) }}
              onClick={() => fileRef.current.click()}
            >
              <div style={{ fontSize: 32 }}>📊</div>
              <p>Click or drag to upload<br />
                <span style={{ fontSize: 11 }}>
                  any image — electrode frame, screenshot, or a figure from a paper
                </span>
              </p>
            </div>
            <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }}
              onChange={e => pickFile(e.target.files[0])} />

            {file && (
              <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text2)' }}>
                <span style={{ fontSize: 14 }}>📄</span>
                <span style={{ color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{file.name}</span>
                <span style={{ marginLeft: 'auto', whiteSpace: 'nowrap' }}>{(file.size / 1024).toFixed(0)} KB</span>
              </div>
            )}

            <div style={{ marginTop: 14 }}>
              <div className="ctrl-section">
                <label className="ctrl-label">Peaks to report <span>{params.n_peaks}</span></label>
                <input type="range" min="5" max="50" step="1" value={params.n_peaks}
                  onChange={e => setP('n_peaks', +e.target.value)} />
              </div>
              <div className="ctrl-section">
                <label className="ctrl-label">Histogram smoothing <span>{params.hist_smooth} bins</span></label>
                <input type="range" min="0.5" max="6" step="0.5" value={params.hist_smooth}
                  onChange={e => setP('hist_smooth', +e.target.value)} />
                <p style={{ fontSize: 11, color: 'var(--text2)', marginTop: 4 }}>
                  Only affects the Fisher information of the histogram. The derivative of a
                  raw 256-bin histogram is mostly counting noise, so it is smoothed first.
                  Shannon entropy is computed on the unsmoothed histogram and does not move.
                </p>
              </div>
              <div className="ctrl-section">
                <label className="ctrl-label" style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'flex-start', cursor: 'pointer' }}>
                  <input type="checkbox" checked={params.mask_timestamp}
                    onChange={e => setP('mask_timestamp', e.target.checked)}
                    style={{ width: 'auto', margin: 0 }} />
                  Ignore burnt-in camera text
                </label>
                <p style={{ fontSize: 11, color: 'var(--text2)', marginTop: 4 }}>
                  The timestamp in the corner has the sharpest edges in the frame. Left in,
                  it dominates every map and every peak.
                </p>
              </div>
            </div>

            <button className="btn primary" style={{ marginTop: 14, width: '100%' }}
              disabled={!file || loading} onClick={analyze}>
              {loading ? 'Computing…' : '▶ Analyze picture'}
            </button>
          </div>

          {result && (
            <div className="card section-gap">
              <div className="card-title">From the histogram</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                {[
                  { label: 'Shannon H (nats)', val: result.shannon_entropy, cls: 'blue' },
                  { label: 'H normalised', val: result.shannon_entropy_norm + ' %', cls: 'blue' },
                  { label: 'Fisher J', val: Number(result.fisher_histogram_J).toExponential(3), cls: 'green' },
                ].map(m => (
                  <div key={m.label} className="metric">
                    <div className="metric-label">{m.label}</div>
                    <div className={`metric-value ${m.cls || ''}`} style={{ fontSize: 15 }}>{m.val}</div>
                  </div>
                ))}
              </div>
              <div className="interp-box" style={{ marginTop: 10 }}>
                Entropy asks how <strong>spread out</strong> the histogram is.
                Fisher J asks how <strong>sharply it changes</strong> along the brightness
                axis. Two different questions about the same histogram — a picture can be
                high in one and low in the other.
              </div>

              <div className="card-title" style={{ marginTop: 18 }}>From the image</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                {[
                  { label: 'Fisher total, x', val: Number(result.fisher_x_total).toExponential(2) },
                  { label: 'Fisher total, y', val: Number(result.fisher_y_total).toExponential(2) },
                  { label: 'x / y ratio', val: result.fisher_xy_ratio ?? '—', cls: 'blue' },
                  { label: 'Mean F', val: Number(result.fisher_mean).toExponential(2) },
                  { label: 'Median F', val: Number(result.fisher_median).toExponential(2) },
                  { label: 'Edge / flat', val: result.edge_flat_ratio ?? '—', cls: 'green' },
                ].map(m => (
                  <div key={m.label} className="metric">
                    <div className="metric-label">{m.label}</div>
                    <div className={`metric-value ${m.cls || ''}`} style={{ fontSize: 15 }}>{m.val}</div>
                  </div>
                ))}
              </div>
              <div className="interp-box" style={{ marginTop: 10 }}>
                <strong>x / y ratio {result.fisher_xy_ratio ?? '—'}</strong> — {ratioNote(result.fisher_xy_ratio)}.<br />
                <strong>Edge / flat {result.edge_flat_ratio ?? '—'}</strong> — {edgeNote(result.edge_flat_ratio)}.
              </div>

              <button className="btn success" style={{ marginTop: 12, width: '100%' }}
                disabled={exporting} onClick={exportResults}>
                {exporting ? 'Preparing…' : '⬇ Export to Excel (histogram + summary + peaks)'}
              </button>
            </div>
          )}
        </div>

        {/* ── right ──────────────────────────────────────────────── */}
        <div style={{ minWidth: 0 }}>
          <div className="card">
            <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
              {maps.map(([id, label]) => (
                <button key={id} className={`btn${view === id ? ' primary' : ''}`} style={{ fontSize: 11 }}
                  onClick={() => setView(id)}>{label}</button>
              ))}
              {result && (
                <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text2)' }}>
                  {result.width} × {result.height} px
                </span>
              )}
            </div>
            <div style={{ minHeight: 300, background: '#000', borderRadius: 6, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {shown
                ? <img src={shown} alt="Fisher map" style={{ width: '100%', maxHeight: 420, objectFit: 'contain' }} />
                : <div style={{ color: 'var(--text2)', textAlign: 'center' }}>
                    <div style={{ fontSize: 40 }}>📊</div>
                    <p style={{ marginTop: 8, fontSize: 13 }}>
                      {file ? 'Press Analyze picture' : 'Upload a picture to begin'}
                    </p>
                  </div>}
            </div>
            {result && view === 'peaks' && (
              <p style={{ fontSize: 11, color: 'var(--text2)', marginTop: 8 }}>
                Green = the peak sits on a brightness edge found independently, so it is a
                real boundary. Blue = it does not, which usually means the division by a
                very small brightness, not a physical structure.
              </p>
            )}
          </div>

          {result && (
            <div className="card section-gap">
              <div className="card-title">Brightness histogram</div>
              <img src={'data:image/png;base64,' + result.histogram_image}
                alt="Brightness histogram" style={{ width: '100%' }} />
            </div>
          )}

          {result && (
            <div className="card section-gap">
              <div className="card-title">Strongest Fisher peaks</div>
              <div style={{ maxHeight: 250, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ color: 'var(--text2)', textAlign: 'right' }}>
                      <th style={{ textAlign: 'left', padding: '4px 6px' }}>#</th>
                      <th style={{ padding: '4px 6px' }}>x</th>
                      <th style={{ padding: '4px 6px' }}>y</th>
                      <th style={{ padding: '4px 6px' }}>× median</th>
                      <th style={{ padding: '4px 6px' }}>I</th>
                      <th style={{ padding: '4px 6px' }}>on edge</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.peaks.map((p, i) => (
                      <tr key={i} style={{ borderTop: '1px solid #30363d', textAlign: 'right' }}>
                        <td style={{ textAlign: 'left', padding: '3px 6px' }}>{i + 1}</td>
                        <td style={{ padding: '3px 6px' }}>{p.x}</td>
                        <td style={{ padding: '3px 6px' }}>{p.y}</td>
                        <td style={{ padding: '3px 6px', color: 'var(--blue)' }}>{p.height_vs_median}</td>
                        <td style={{ padding: '3px 6px' }}>{p.brightness}</td>
                        <td style={{ padding: '3px 6px', color: p.on_edge ? '#3fb950' : '#d29922' }}>
                          {p.on_edge ? 'yes' : 'no'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p style={{ fontSize: 11, color: 'var(--text2)', marginTop: 8 }}>
                <strong>× median</strong> is the significance: how many times the frame's own
                median Fisher value that peak reaches. A peak at 3 × median is unremarkable;
                one at 50 × is a genuine boundary. A high value with <em>on edge = no</em> is
                an artefact of dividing by a small brightness, not a structure.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
