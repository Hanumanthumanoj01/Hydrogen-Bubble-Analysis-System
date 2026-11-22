import { useRef, useState } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { exportToExcel } from '../utils/export'

export default function ImageAnalyzer() {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [overlay, setOverlay] = useState(null)
  const [heatmap, setHeatmap] = useState(null)
  const [histogram, setHistogram] = useState(null)
  const [view, setView] = useState('overlay') // overlay | heatmap | histogram
  const [file, setFile] = useState(null)
  const fileRef = useRef()

  const [params, setParams] = useState({
    current_density: 200,
    electrode_type: 'gde',
    cathode_area: 2.54,
    sensitivity: 'medium',
    detector: 'primary',
    min_pct: 3,
  })

  // Nothing is analysed until the button is pressed. Picking a file only
  // remembers it, so the settings underneath can be set first.
  const pickFile = (f) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setOverlay(null); setHeatmap(null); setHistogram(null)
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('current_density', params.current_density)
    fd.append('electrode_type', params.electrode_type)
    fd.append('cathode_area', params.cathode_area)
    fd.append('sensitivity', params.sensitivity)
    fd.append('detector', params.detector)
    fd.append('min_pct', params.min_pct)
    try {
      const { data } = await axios.post('/analyze', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setResult(data)
      setOverlay('data:image/png;base64,' + data.overlay_image)
      setHeatmap('data:image/png;base64,' + data.heatmap_image)
      setHistogram('data:image/png;base64,' + data.histogram_image)
      setView('overlay')
    } catch (e) {
      const msg = e?.response?.data?.detail
      alert(msg || 'Backend not running — start the FastAPI backend first.\n\nuvicorn main:app --reload --port 8000')
    }
    setLoading(false)
  }

  const handleDrop = e => {
    e.preventDefault(); setDragging(false)
    pickFile(e.dataTransfer.files[0])
  }

  const exportResults = () => {
    if (!result) return
    exportToExcel(result.bubbles.map((b, i) => ({
      index: i + 1,
      x_px: b.x, y_px: b.y,
      r_px: b.r_px,
      diameter_mm: b.diameter_mm,
      curvature_1_per_mm: b.curvature,
    })), ['index','x_px','y_px','r_px','diameter_mm','curvature_1_per_mm'], 'bubble_analysis.xlsx',
    {
      sheet2: {
        name: 'Summary',
        data: [
          { metric: 'Source file', value: file ? file.name : '' },
          { metric: 'Detection mode', value: params.detector === 'primary' ? 'Main bubbles' : 'Every bubble' },
          { metric: 'Shannon entropy H', value: result.shannon_entropy },
          { metric: 'H_norm (%)', value: result.shannon_entropy_norm },
          { metric: 'Bubble count', value: result.bubble_count },
          { metric: 'Mean diameter (mm)', value: result.mean_diameter_mm },
          { metric: 'Mean curvature (1/mm)', value: result.mean_curvature_1_per_mm },
          { metric: 'Surface coverage (%)', value: result.surface_coverage_pct },
          { metric: 'Nucleation sites', value: result.nucleation_sites },
          { metric: 'H2 production (mL/min)', value: result.H2_production_mL_per_min },
          { metric: 'Electrode type', value: result.electrode_type },
          { metric: 'Current density (mA/cm²)', value: result.current_density },
          { metric: 'Cathode area (cm²)', value: result.cathode_area },
        ],
        cols: ['metric','value']
      }
    })
  }

  const setP = (k, v) => setParams(p => ({ ...p, [k]: v }))

  const interpText = (hn) => {
    if (hn < 70) return 'Low entropy (H_norm < 70 %): sparse bubble activity — electrode near thermoneutral voltage.'
    if (hn < 85) return 'Moderate entropy (70–85 %): typical GDE behaviour at intermediate current density.'
    return 'High entropy (> 85 %): dense, chaotic bubble layer — high current density or froth formation (mesh electrode).'
  }

  return (
    <div className="panel">
      <div className="grid-2-equal">
        {/* Left col */}
        <div>
          <div className="card">
            <div className="card-title">Upload microscope image</div>
            <div
              className={`drop-zone${dragging ? ' dragover' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current.click()}
            >
              <div style={{ fontSize: 32 }}>🔬</div>
              <p>Click or drag to upload<br /><span style={{ fontSize: 11 }}>JPG / PNG — e.g. bubble_density_100mA.jpg</span></p>
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
                <select value={params.detector} onChange={e => setP('detector', e.target.value)}>
                  <option value="primary">Main bubbles — one circle per bubble</option>
                  <option value="standard">Every bubble, froth included</option>
                </select>
              </div>
              {params.detector === 'primary' && (
                <div className="ctrl-section">
                  <label className="ctrl-label">
                    Smallest bubble to detect <span>{params.min_pct} % of image height</span>
                  </label>
                  <input type="range" min="2" max="25" step="1" value={params.min_pct}
                    onChange={e => setP('min_pct', +e.target.value)} />
                  <p style={{ fontSize: 11, color: 'var(--text2)', marginTop: 4 }}>
                    Raise this if you are looking at a zoomed-in view and small
                    marks on the plate are being circled. 4 % suits a full frame;
                    8–12 % suits a close-up of two or three bubbles.
                  </p>
                </div>
              )}
              {params.detector === 'standard' && (
                <div className="ctrl-section">
                  <label className="ctrl-label">Detection sensitivity</label>
                  <select value={params.sensitivity} onChange={e => setP('sensitivity', e.target.value)}>
                    <option value="low">Low (fewer false positives)</option>
                    <option value="medium">Medium (balanced)</option>
                    <option value="high">High (more detections)</option>
                  </select>
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
            </div>

            <button className="btn primary" style={{ marginTop: 14, width: '100%' }}
              disabled={!file || loading} onClick={analyze}>
              {loading ? 'Analyzing…' : '▶ Analyze image'}
            </button>
            {!file && (
              <p style={{ fontSize: 11, color: 'var(--text2)', marginTop: 6, textAlign: 'center' }}>
                Choose an image, set the current density, then press Analyze.
              </p>
            )}
          </div>

          {/* Metrics */}
          {result && (
            <div className="card section-gap">
              <div className="card-title">Analysis results</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                {[
                  { label: 'Bubbles detected', val: result.bubble_count, cls: 'blue' },
                  { label: 'Shannon H', val: result.shannon_entropy, cls: 'blue' },
                  { label: 'H_norm %', val: result.shannon_entropy_norm + '%', cls: 'blue' },
                  { label: 'Mean diameter', val: result.mean_diameter_mm + ' mm' },
                  { label: 'Mean curvature K', val: result.mean_curvature_1_per_mm + ' /mm' },
                  { label: 'Coverage', val: result.surface_coverage_pct + '%' },
                  { label: 'Nucleation sites', val: result.nucleation_sites },
                  { label: 'H₂ estimate', val: result.H2_production_mL_per_min + ' mL/min', cls: 'green' },
                  { label: 'Density /cm²', val: result.bubble_density_per_cm2 },
                ].map(m => (
                  <div key={m.label} className="metric">
                    <div className="metric-label">{m.label}</div>
                    <div className={`metric-value ${m.cls || ''}`} style={{ fontSize: 15 }}>{m.val}</div>
                  </div>
                ))}
              </div>

              {/* Entropy bar */}
              <div style={{ marginTop: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text2)', marginBottom: 4 }}>
                  <span>Entropy level</span><span>{result.shannon_entropy_norm}%</span>
                </div>
                <div className="entropy-bar-wrap">
                  <div className="entropy-bar" style={{ width: result.shannon_entropy_norm + '%' }} />
                </div>
              </div>

              <div className="interp-box">{interpText(result.shannon_entropy_norm)}</div>

              <button className="btn success" style={{ marginTop: 12, width: '100%' }} onClick={exportResults}>
                ⬇ Export to Excel (bubbles + summary)
              </button>
            </div>
          )}
        </div>

        {/* Right col */}
        <div>
          {/* Image viewer */}
          <div className="card">
            <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center' }}>
              {[['overlay','🔴 Detection overlay'],['heatmap','🌡 Density heatmap'],['histogram','📊 Pixel histogram']].map(([id, label]) => (
                <button key={id} className={`btn${view === id ? ' primary' : ''}`} style={{ fontSize: 12 }}
                  onClick={() => setView(id)}>{label}</button>
              ))}
              {result && file && (
                <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 220 }}>
                  {file.name}
                </span>
              )}
            </div>
            <div style={{ minHeight: 300, background: '#000', borderRadius: 6, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {view === 'overlay' && overlay && <img src={overlay} alt="Bubble detection overlay" style={{ width: '100%', maxHeight: 400, objectFit: 'contain' }} />}
              {view === 'heatmap' && heatmap && <img src={heatmap} alt="Bubble density heatmap" style={{ width: '100%', maxHeight: 400, objectFit: 'contain' }} />}
              {view === 'histogram' && histogram && <img src={histogram} alt="Pixel brightness histogram" style={{ width: '100%', maxHeight: 400, objectFit: 'contain' }} />}
              {!result && <div style={{ color: 'var(--text2)', textAlign: 'center' }}><div style={{ fontSize: 40 }}>🔬</div><p style={{ marginTop: 8, fontSize: 13 }}>{file ? 'Press Analyze image' : 'Upload an image to begin'}</p></div>}
            </div>
          </div>

          {/* Size distribution */}
          {result && (
            <div className="card section-gap">
              <div className="card-title">Bubble size distribution (Wagner et al. 2025, Fig. 4 bins)</div>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={result.size_distribution.labels.map((l, i) => ({ bin: l, count: result.size_distribution.counts[i] }))}>
                  <CartesianGrid stroke="#30363d" strokeDasharray="3 3" />
                  <XAxis dataKey="bin" tick={{ fontSize: 10, fill: '#8b949e' }} label={{ value: 'Diameter (mm)', position: 'insideBottom', offset: -2, fill: '#8b949e', fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10, fill: '#8b949e' }} />
                  <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', fontSize: 12 }} />
                  <Bar dataKey="count" fill="#58a6ff" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
