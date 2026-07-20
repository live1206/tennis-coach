import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import type { LoadedAnalysis } from '../../shared/analysis'
import { expandEvidenceReferences } from '../evidenceFormatting'
import { useCopy } from '../i18n'
import { renderMarkdown } from '../markdown'

interface Props {
  loaded: LoadedAnalysis
  languageSwitch: ReactNode
  onBack: () => void
}

export default function AIAnalysisScreen({ loaded, languageSwitch, onBack }: Props) {
  const copy = useCopy()
  const startedRef = useRef(false)
  const [running, setRunning] = useState(true)
  const [result, setResult] = useState('')
  const [error, setError] = useState('')
  const { analysis } = loaded

  const runCloudAnalysis = useCallback(async () => {
    setRunning(true)
    setError('')
    setResult('')
    try {
      const response = await window.api.runCloudAIAnalysis(
        loaded.videoPath,
        loaded.evidenceId,
        copy.aiAnalysis.defaultQuestion,
      )
      if (response.error) setError(response.error)
      else setResult(expandEvidenceReferences(response.output ?? '', analysis))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : copy.aiAnalysis.cloudUnknownError)
    } finally {
      setRunning(false)
    }
  }, [analysis, copy.aiAnalysis.cloudUnknownError, copy.aiAnalysis.defaultQuestion, loaded.evidenceId, loaded.videoPath])

  useEffect(() => {
    if (startedRef.current) return
    startedRef.current = true
    void runCloudAnalysis()
  }, [runCloudAnalysis])

  const cancelAnalysis = async () => {
    await window.api.cancelCloudAIAnalysis()
  }

  const returnToReview = async () => {
    if (running) await cancelAnalysis()
    onBack()
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--color-bg)', color: 'var(--color-text)' }}>
      <header style={headerStyle}>
        <div>
          <div style={eyebrowStyle}>{copy.aiAnalysis.eyebrow}</div>
          <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 24 }}>
            {copy.aiAnalysis.title}
          </h1>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {languageSwitch}
          <button style={secondaryButton} onClick={returnToReview}>{copy.aiAnalysis.back}</button>
        </div>
      </header>

      <main style={mainStyle}>
        <section style={{ ...cardStyle, ...scrollPanelStyle }}>
          <h2 style={sectionTitle}>{copy.aiAnalysis.evidenceTitle}</h2>
          <Metric label={copy.aiAnalysis.segments} value={analysis.source?.segment_count ?? analysis.segments.length} />
          <Metric label={copy.aiAnalysis.players} value={Object.keys(analysis.players).length} />

          <h3 style={subheading}>{copy.aiAnalysis.actualResults}</h3>
          <Metric label={copy.aiAnalysis.audioHits} value={analysis.data_quality.audio?.hit_count ?? 0} />
          <Metric
            label={copy.aiAnalysis.classifiedShots}
            value={`${analysis.data_quality.shots?.classified_count ?? 0} / ${analysis.data_quality.shots?.candidate_count ?? 0}`}
          />
          <Metric
            label={copy.aiAnalysis.ballDetections}
            value={analysis.data_quality.ball?.detected_visible_count ?? analysis.data_quality.ball?.visible_count ?? 0}
          />
          <Metric
            label={copy.aiAnalysis.ballVisibility}
            value={formatPercentage(analysis.data_quality.ball?.detected_visible_ratio ?? analysis.data_quality.ball?.visible_ratio)}
          />

          <h3 style={subheading}>{copy.aiAnalysis.playerResults}</h3>
          {Object.entries(analysis.players).map(([playerId, player]) => (
            <div key={playerId} style={playerCardStyle}>
              <strong style={playerTitleStyle}>{formatPlayerName(playerId)}</strong>
              <Metric label={copy.aiAnalysis.forehands} value={player.shot_counts?.forehand ?? 0} compact />
              <Metric label={copy.aiAnalysis.backhands} value={player.shot_counts?.backhand ?? 0} compact />
              <Metric label={copy.aiAnalysis.courtMovement} value={formatDistance(player.total_court_movement_normalized)} compact />
              <Metric label={copy.aiAnalysis.trajectorySamples} value={player.trajectory_samples ?? 0} compact />
            </div>
          ))}
        </section>

        <section style={{ ...cardStyle, ...cloudCardStyle }}>
          <h2 style={sectionTitle}>{copy.aiAnalysis.askTitleCloud}</h2>
          <p style={mutedStyle}>{copy.aiAnalysis.privacyCloud}</p>
          {running && <p style={runningStyle}>{copy.aiAnalysis.cloudRunning}</p>}
          {running && <button onClick={cancelAnalysis} style={{ ...secondaryButton, marginTop: 12 }}>{copy.aiAnalysis.cancel}</button>}

          {error && <div style={errorStyle}>{error}</div>}
          {error && !running && (
            <button onClick={() => void runCloudAnalysis()} style={{ ...secondaryButton, marginTop: 12 }}>
              {copy.aiAnalysis.cloudRetry}
            </button>
          )}
          {result && (
            <div style={resultWrapperStyle}>
              <h3 style={subheading}>{copy.aiAnalysis.resultTitle}</h3>
              <div style={resultStyle}>{renderMarkdown(result)}</div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

function Metric({ label, value, compact = false }: { label: string; value: number | string; compact?: boolean }) {
  return <div style={{ display: 'flex', justifyContent: 'space-between', padding: compact ? '5px 0' : '9px 0', borderBottom: '1px solid var(--color-border)' }}>
    <span style={mutedStyle}>{label}</span><strong>{value}</strong>
  </div>
}

function formatPercentage(value: number | null | undefined): string {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'
}

function formatDistance(value: number | undefined): string {
  return typeof value === 'number' ? value.toFixed(2) : '—'
}

function formatPlayerName(playerId: string): string {
  return playerId.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

const headerStyle: CSSProperties = { padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--color-border)', background: 'var(--color-surface)', WebkitAppRegion: 'drag' } as CSSProperties
const mainStyle: CSSProperties = { flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: 'minmax(280px, 0.8fr) minmax(420px, 1.2fr)', gap: 20, padding: 24, overflow: 'hidden' }
const cardStyle: CSSProperties = { background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-lg)', padding: 24, minWidth: 0 }
const scrollPanelStyle: CSSProperties = { minHeight: 0, overflowY: 'auto', scrollbarGutter: 'stable' }
const eyebrowStyle: CSSProperties = { color: 'var(--color-accent)', fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700, letterSpacing: '0.16em', textTransform: 'uppercase', marginBottom: 5 }
const sectionTitle: CSSProperties = { fontFamily: 'var(--font-display)', fontSize: 18, margin: '0 0 14px' }
const subheading: CSSProperties = { fontFamily: 'var(--font-display)', fontSize: 13, margin: '22px 0 8px', textTransform: 'uppercase', letterSpacing: '0.05em' }
const mutedStyle: CSSProperties = { color: 'var(--color-text-secondary)', fontSize: 12, lineHeight: 1.5 }
const playerCardStyle: CSSProperties = { marginTop: 10, padding: '10px 12px', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', background: 'rgba(255,255,255,0.55)' }
const playerTitleStyle: CSSProperties = { display: 'block', marginBottom: 3, fontFamily: 'var(--font-display)', fontSize: 13 }
const runningStyle: CSSProperties = { marginTop: 20, color: 'var(--color-accent)', fontFamily: 'var(--font-display)', fontWeight: 800 }
const secondaryButton: CSSProperties = { padding: '8px 12px', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', background: 'var(--color-surface)', cursor: 'pointer', WebkitAppRegion: 'no-drag' } as CSSProperties
const errorStyle: CSSProperties = { marginTop: 16, padding: 12, borderRadius: 'var(--radius-md)', color: 'var(--color-danger)', background: 'rgba(196,91,91,0.08)', whiteSpace: 'pre-wrap', fontSize: 12 }
const cloudCardStyle: CSSProperties = { display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }
const resultWrapperStyle: CSSProperties = { marginTop: 20, flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }
const resultStyle: CSSProperties = { margin: 0, padding: 16, borderRadius: 'var(--radius-md)', background: '#fff', border: '1px solid var(--color-border)', overflowWrap: 'anywhere', fontFamily: 'var(--font-body)', fontSize: 13, lineHeight: 1.6, flex: 1, minHeight: 0, overflowY: 'auto' }
