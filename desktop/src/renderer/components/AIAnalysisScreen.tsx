import { useState, type CSSProperties, type ReactNode } from 'react'
import type { LoadedAnalysis } from '../../shared/analysis'
import { useCopy } from '../i18n'

interface Props {
  loaded: LoadedAnalysis
  languageSwitch: ReactNode
  onBack: () => void
}

export default function AIAnalysisScreen({ loaded, languageSwitch, onBack }: Props) {
  const copy = useCopy()
  const [question, setQuestion] = useState(copy.aiAnalysis.defaultQuestion)
  const [cloudModel, setCloudModel] = useState('gpt-5.6-sol')
  const [localModel, setLocalModel] = useState('qwen2.5-0.5b')
  const [provider, setProvider] = useState<'cloud' | 'local'>('cloud')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState('')
  const [error, setError] = useState('')
  const { analysis } = loaded
  const model = provider === 'cloud' ? cloudModel : localModel
  const setModel = provider === 'cloud' ? setCloudModel : setLocalModel

  const runAnalysis = async () => {
    setRunning(true)
    setError('')
    setResult('')
    try {
      const response = provider === 'cloud'
        ? await window.api.runCloudAIAnalysis(
          loaded.videoPath,
          loaded.evidenceId,
          question,
          model,
        )
        : await window.api.runLocalAIAnalysis(
          loaded.videoPath,
          loaded.evidenceId,
          question,
          model,
        )
      if (response.error) setError(response.error)
      else setResult(response.output ?? '')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : copy.aiAnalysis.unknownError)
    } finally {
      setRunning(false)
    }
  }

  const cancelAnalysis = async () => {
    await window.api.cancelAIAnalysis()
  }

  const returnToWelcome = async () => {
    if (running) await cancelAnalysis()
    onBack()
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--color-bg)', color: 'var(--color-text)' }}>
      <header style={headerStyle}>
        <div>
          <div style={eyebrowStyle}>{copy.aiAnalysis.eyebrow}</div>
          <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 24 }}>
            {copy.aiAnalysis.title}
          </h1>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {languageSwitch}
          <button style={secondaryButton} onClick={returnToWelcome}>{copy.aiAnalysis.back}</button>
        </div>
      </header>

      <main style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 0.8fr) minmax(420px, 1.2fr)', gap: 20, padding: 24 }}>
        <section style={cardStyle}>
          <h2 style={sectionTitle}>{copy.aiAnalysis.evidenceTitle}</h2>
          <Metric label={copy.aiAnalysis.segments} value={analysis.source?.segment_count ?? analysis.segments.length} />
          <Metric label={copy.aiAnalysis.players} value={Object.keys(analysis.players).length} />

          <h3 style={subheading}>{copy.aiAnalysis.supported}</h3>
          <ul style={listStyle}>
            {analysis.analysis_capabilities.supported.map((capability) => <li key={capability}>{capability}</li>)}
          </ul>
        </section>

        <section style={cardStyle}>
          <h2 style={sectionTitle}>
            {provider === 'cloud' ? copy.aiAnalysis.askTitleCloud : copy.aiAnalysis.askTitleLocal}
          </h2>
          <p style={mutedStyle}>
            {provider === 'cloud' ? copy.aiAnalysis.privacyCloud : copy.aiAnalysis.privacyLocal}
          </p>
          <label style={labelStyle}>
            {copy.aiAnalysis.provider}
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                type="button"
                disabled={running}
                onClick={() => setProvider('cloud')}
                style={provider === 'cloud' ? selectedProviderButton : providerButton}
              >
                {copy.aiAnalysis.providerCloud}
              </button>
              <button
                type="button"
                disabled={running}
                onClick={() => setProvider('local')}
                style={provider === 'local' ? selectedProviderButton : providerButton}
              >
                {copy.aiAnalysis.providerLocal}
              </button>
            </div>
          </label>
          <label style={labelStyle}>
            {provider === 'cloud' ? copy.aiAnalysis.modelCloud : copy.aiAnalysis.modelLocal}
            <input value={model} onChange={(event) => setModel(event.target.value)} style={inputStyle} />
          </label>
          <label style={labelStyle}>
            {copy.aiAnalysis.question}
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={5} style={{ ...inputStyle, resize: 'vertical' }} />
          </label>
          <button disabled={running || !question.trim() || !model.trim()} onClick={runAnalysis} style={primaryButton}>
            {running
              ? copy.aiAnalysis.running
              : (provider === 'cloud' ? copy.aiAnalysis.runCloud : copy.aiAnalysis.runLocal)}
          </button>
          {running && <button onClick={cancelAnalysis} style={{ ...secondaryButton, marginLeft: 8 }}>{copy.aiAnalysis.cancel}</button>}

          {error && <div style={errorStyle}>{error}</div>}
          {result && (
            <div style={{ marginTop: 20 }}>
              <h3 style={subheading}>{copy.aiAnalysis.resultTitle}</h3>
              <pre style={resultStyle}>{result}</pre>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: number }) {
  return <div style={{ display: 'flex', justifyContent: 'space-between', padding: '9px 0', borderBottom: '1px solid var(--color-border)' }}>
    <span style={mutedStyle}>{label}</span><strong>{value}</strong>
  </div>
}

const headerStyle: CSSProperties = { padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--color-border)', background: 'var(--color-surface)', WebkitAppRegion: 'drag' } as CSSProperties
const cardStyle: CSSProperties = { background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-lg)', padding: 24, minWidth: 0 }
const eyebrowStyle: CSSProperties = { color: 'var(--color-accent)', fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700, letterSpacing: '0.16em', textTransform: 'uppercase', marginBottom: 5 }
const sectionTitle: CSSProperties = { fontFamily: 'var(--font-display)', fontSize: 18, margin: '0 0 14px' }
const subheading: CSSProperties = { fontFamily: 'var(--font-display)', fontSize: 13, margin: '22px 0 8px', textTransform: 'uppercase', letterSpacing: '0.05em' }
const mutedStyle: CSSProperties = { color: 'var(--color-text-secondary)', fontSize: 12, lineHeight: 1.5 }
const listStyle: CSSProperties = { margin: 0, paddingLeft: 20, color: 'var(--color-text-secondary)', fontSize: 12, lineHeight: 1.6 }
const labelStyle: CSSProperties = { display: 'flex', flexDirection: 'column', gap: 7, fontSize: 12, fontWeight: 700, marginTop: 16 }
const inputStyle: CSSProperties = { padding: '10px 12px', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', background: '#fff', color: 'var(--color-text)', font: 'inherit' }
const providerButton: CSSProperties = { padding: '8px 12px', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', background: 'var(--color-surface)', color: 'var(--color-text)', cursor: 'pointer', fontWeight: 700 }
const selectedProviderButton: CSSProperties = { ...providerButton, border: '1px solid var(--color-accent)', background: 'rgba(204,78,14,0.08)', color: 'var(--color-accent)' }
const primaryButton: CSSProperties = { marginTop: 18, padding: '11px 18px', border: 0, borderRadius: 'var(--radius-md)', background: 'var(--color-accent)', color: '#fff', cursor: 'pointer', fontWeight: 700 }
const secondaryButton: CSSProperties = { padding: '8px 12px', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', background: 'var(--color-surface)', cursor: 'pointer', WebkitAppRegion: 'no-drag' } as CSSProperties
const errorStyle: CSSProperties = { marginTop: 16, padding: 12, borderRadius: 'var(--radius-md)', color: 'var(--color-danger)', background: 'rgba(196,91,91,0.08)', whiteSpace: 'pre-wrap', fontSize: 12 }
const resultStyle: CSSProperties = { margin: 0, padding: 16, borderRadius: 'var(--radius-md)', background: '#fff', border: '1px solid var(--color-border)', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', fontFamily: 'var(--font-body)', fontSize: 13, lineHeight: 1.6 }
