const PATH_REFERENCE = /^(?:\$\.?)?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+$/
const INLINE_PATH_REFERENCE = /`?((?:\$\.?)?(?:schema|source|data_quality|analysis_capabilities|target_player|players|segments)(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+)`?(?=[\s,;:)\]}.]|$)/g

export function expandEvidenceReferences(text: string, analysis: object): string {
  const grouped = expandReferenceGroup(
    expandReferenceGroup(
      text,
      analysis,
      /\[((?:[^\]\n]|\](?=\.))*)\]/g,
      '[',
      ']',
    ),
    analysis,
    /\(([^()\n]+)\)/g,
    '(',
    ')',
  )
  return grouped.replace(
    INLINE_PATH_REFERENCE,
    (original, path: string) => {
      const value = resolveEvidencePath(analysis, path)
      return value.found
        ? `${formatEvidenceLabel(path)}: ${formatEvidenceValue(value.value, path)}`
        : original
    },
  )
}

function expandReferenceGroup(
  text: string,
  analysis: object,
  pattern: RegExp,
  opening: string,
  closing: string,
): string {
  return text.replace(pattern, (original, content: string) => {
    const paths = content
      .split(/[;,]/)
      .map((path) => path.trim().replace(/^`|`$/g, '').trim())
    if (paths.length === 0 || paths.some((path) => !PATH_REFERENCE.test(path))) {
      return original
    }

    const evidence = paths.map((path) => {
      const value = resolveEvidencePath(analysis, path)
      return value.found
        ? `${formatEvidenceLabel(path)}: ${formatEvidenceValue(value.value, path)}`
        : null
    })
    return evidence.every((item) => item !== null)
      ? `${opening}${evidence.join('; ')}${closing}`
      : original
  })
}

function resolveEvidencePath(root: object, path: string): { found: boolean; value?: unknown } {
  const tokens = path.replace(/^\$\.?/, '').match(/[A-Za-z_][A-Za-z0-9_]*|\d+/g) ?? []
  let current: unknown = root
  for (const token of tokens) {
    if (Array.isArray(current)) {
      const index = Number(token)
      if (!Number.isInteger(index) || index < 0 || index >= current.length) {
        return { found: false }
      }
      current = current[index]
    } else if (isRecord(current) && token in current) {
      current = current[token]
    } else {
      return { found: false }
    }
  }
  return { found: true, value: current }
}

function formatEvidenceLabel(path: string): string {
  return path
    .replace(/^\$\.?/, '')
    .replace(/\[(\d+)\]/g, ' $1')
    .split('.')
    .map((part) => part.replace(/_/g, ' '))
    .join(' / ')
}

function formatEvidenceValue(value: unknown, path = ''): string {
  if (Array.isArray(value)) {
    if (path.endsWith('.shots')) {
      return formatShotSummary(value)
    }
    if (value.every(isRecord)) {
      return `${value.length} records`
    }
    return value.map((item) => formatEvidenceValue(item)).join(', ')
  }
  if (isRecord(value)) {
    return Object.entries(value)
      .map(([key, item]) => `${key.replace(/_/g, ' ')} ${formatEvidenceValue(item)}`)
      .join(', ')
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(3)))
  }
  if (value === null) return 'unknown'
  return String(value)
}

function formatShotSummary(value: unknown[]): string {
  const shots = value.filter(isRecord)
  const classifications = shots.reduce<Record<string, number>>((counts, shot) => {
    const classification = shot.classification
    if (classification === 'forehand' || classification === 'backhand') {
      counts[classification] = (counts[classification] ?? 0) + 1
    }
    return counts
  }, {})
  const classified = Object.values(classifications).reduce((total, count) => total + count, 0)
  const details = Object.entries(classifications)
    .map(([classification, count]) => `${classification} ${count}`)
    .join(', ')
  return `${shots.length} candidates (${classified} classified${details ? `: ${details}` : ''})`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
