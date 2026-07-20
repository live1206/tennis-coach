const PATH_REFERENCE = /^\$?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+$/

export function expandEvidenceReferences(text: string, analysis: object): string {
  return expandReferenceGroup(
    expandReferenceGroup(text, analysis, /\[([^\]\n]+)\]/g, '[', ']'),
    analysis,
    /\(([^()\n]+)\)/g,
    '(',
    ')',
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
      .split(';')
      .map((path) => path.trim().replace(/^`|`$/g, '').trim())
    if (paths.length === 0 || paths.some((path) => !PATH_REFERENCE.test(path))) {
      return original
    }

    const evidence = paths.map((path) => {
      const value = resolveEvidencePath(analysis, path)
      return value.found
        ? `${formatEvidenceLabel(path)}: ${formatEvidenceValue(value.value)}`
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

function formatEvidenceValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map(formatEvidenceValue).join(', ')
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
