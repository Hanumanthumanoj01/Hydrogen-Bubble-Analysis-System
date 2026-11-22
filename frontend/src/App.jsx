import { useState } from 'react'
import ImageAnalyzer from './pages/ImageAnalyzer'
import CameraAnalyzer from './pages/CameraAnalyzer'
import VideoAnalyzer from './pages/VideoAnalyzer'
import FisherAnalyzer from './pages/FisherAnalyzer'

const TABS = [
  { id: 'img', label: '🔬 Tab 1 — Image Analyzer' },
  { id: 'cam', label: '📷 Tab 2 — Camera / Video' },
  { id: 'vid', label: '🎬 Tab 3 — Video Analyzer' },
  { id: 'fis', label: '📊 Tab 4 — Histogram / Fisher' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('img')

  return (
    <div className="app">
      <div className="topbar">
        <span className="atom-icon">⚗️</span>
        <div>
          <h1>Hydrogen Bubble Analysis System</h1>
          <p>Entropy-Information Approach · Fuel Cell Laboratory · Frankfurt UAS</p>
        </div>
        <span className="badge" style={{ marginLeft: 'auto' }}>M.Eng. IT · Manoj Hanumanthu</span>
      </div>

      <div className="tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab-btn${activeTab === t.id ? ' active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ flex: 1 }}>
        {activeTab === 'img' && <ImageAnalyzer />}
        {activeTab === 'cam' && <CameraAnalyzer />}
        {activeTab === 'vid' && <VideoAnalyzer />}
        {activeTab === 'fis' && <FisherAnalyzer />}
      </div>
    </div>
  )
}
