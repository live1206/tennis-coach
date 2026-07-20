import type { CSSProperties, ReactNode } from 'react'

const INLINE_PATTERN = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/

function renderInline(text: string): ReactNode[] {
  const parts: ReactNode[] = []
  let remaining = text
  let key = 0
  while (remaining.length > 0) {
    const match = INLINE_PATTERN.exec(remaining)
    if (!match || match.index === undefined) {
      parts.push(remaining)
      break
    }
    if (match.index > 0) parts.push(remaining.slice(0, match.index))
    const token = match[0]
    if (token.startsWith('**')) {
      parts.push(<strong key={key++}>{token.slice(2, -2)}</strong>)
    } else if (token.startsWith('`')) {
      parts.push(<code key={key++} style={inlineCodeStyle}>{token.slice(1, -1)}</code>)
    } else {
      parts.push(<em key={key++}>{token.slice(1, -1)}</em>)
    }
    remaining = remaining.slice(match.index + token.length)
  }
  return parts
}

export function renderMarkdown(text: string): ReactNode {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const blocks: ReactNode[] = []
  let blockKey = 0

  let paragraphBuffer: string[] = []
  let listBuffer: { ordered: boolean; items: string[] } | null = null
  let codeBuffer: string[] | null = null

  const flushParagraph = () => {
    if (paragraphBuffer.length === 0) return
    blocks.push(<p key={blockKey++} style={paragraphStyle}>{renderInline(paragraphBuffer.join(' '))}</p>)
    paragraphBuffer = []
  }

  const flushList = () => {
    if (!listBuffer) return
    const items = listBuffer.items.map((item, idx) => <li key={idx}>{renderInline(item)}</li>)
    blocks.push(
      listBuffer.ordered
        ? <ol key={blockKey++} style={listStyle}>{items}</ol>
        : <ul key={blockKey++} style={listStyle}>{items}</ul>,
    )
    listBuffer = null
  }

  for (const rawLine of lines) {
    if (codeBuffer !== null) {
      if (/^\s*```/.test(rawLine)) {
        blocks.push(<pre key={blockKey++} style={codeBlockStyle}><code>{codeBuffer.join('\n')}</code></pre>)
        codeBuffer = null
      } else {
        codeBuffer.push(rawLine)
      }
      continue
    }

    if (/^\s*```/.test(rawLine)) {
      flushParagraph()
      flushList()
      codeBuffer = []
      continue
    }

    const headingMatch = /^(#{1,6})\s+(.*)$/.exec(rawLine)
    if (headingMatch) {
      flushParagraph()
      flushList()
      const rawLevel = headingMatch[1].length
      const level = Math.min(rawLevel + 3, 6)
      const HeadingTag = `h${level}` as 'h4' | 'h5' | 'h6'
      blocks.push(
        <HeadingTag key={blockKey++} style={headingStyleForLevel(rawLevel)}>
          {renderInline(headingMatch[2])}
        </HeadingTag>,
      )
      continue
    }

    const bulletMatch = /^\s*[-*]\s+(.*)$/.exec(rawLine)
    const numberedMatch = /^\s*\d+[.)]\s+(.*)$/.exec(rawLine)
    if (bulletMatch || numberedMatch) {
      flushParagraph()
      const ordered = Boolean(numberedMatch)
      const content = (bulletMatch ?? numberedMatch)![1]
      if (!listBuffer || listBuffer.ordered !== ordered) {
        flushList()
        listBuffer = { ordered, items: [] }
      }
      listBuffer.items.push(content)
      continue
    }

    if (rawLine.trim() === '') {
      flushParagraph()
      flushList()
      continue
    }

    paragraphBuffer.push(rawLine.trim())
  }

  flushParagraph()
  flushList()
  if (codeBuffer !== null && codeBuffer.length > 0) {
    blocks.push(<pre key={blockKey++} style={codeBlockStyle}><code>{codeBuffer.join('\n')}</code></pre>)
  }

  return <>{blocks}</>
}

const paragraphStyle: CSSProperties = { margin: '0 0 12px', lineHeight: 1.6 }
const listStyle: CSSProperties = { margin: '0 0 12px', paddingLeft: 22, lineHeight: 1.6 }
const inlineCodeStyle: CSSProperties = { fontFamily: 'var(--font-mono)', background: 'rgba(0,0,0,0.06)', borderRadius: 4, padding: '1px 5px', fontSize: '0.92em' }
const codeBlockStyle: CSSProperties = { margin: '0 0 12px', padding: 12, borderRadius: 'var(--radius-md)', background: 'rgba(0,0,0,0.06)', overflowX: 'auto', fontFamily: 'var(--font-mono)', fontSize: 12.5 }

function headingStyleForLevel(level: number): CSSProperties {
  const base: CSSProperties = {
    fontFamily: 'var(--font-display)',
    fontWeight: 800,
    lineHeight: 1.3,
    color: 'var(--color-text)',
  }
  if (level <= 1) return { ...base, margin: '4px 0 14px', fontSize: 26 }
  if (level === 2) return { ...base, margin: '24px 0 12px', fontSize: 21, color: 'var(--color-accent)' }
  return { ...base, margin: '18px 0 8px', fontSize: 15, textTransform: 'uppercase', letterSpacing: '0.03em' }
}
