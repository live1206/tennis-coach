export interface TennisAnalysis {
  schema: {
    name: 'tennis-coach-analysis'
    version: number
  }
  source?: {
    segment_count?: number
    start?: number | null
    end?: number | null
    duration?: number | null
  }
  data_quality: {
    warnings: string[]
    [key: string]: unknown
  }
  analysis_capabilities: {
    supported: string[]
    unsupported: string[]
  }
  target_player?: {
    player_id?: string | null
    confidence?: number | null
    reason?: string | null
  }
  players: Record<string, {
    trajectory_samples?: number
    shot_counts?: Record<string, number>
    shot_role_counts?: Record<string, number>
    shot_outcome_counts?: Record<string, number>
    [key: string]: unknown
  }>
  segments: unknown[]
}

export interface LoadedAnalysis {
  path: string
  analysis: TennisAnalysis
}

export function validateTennisAnalysis(value: unknown): TennisAnalysis {
  if (!isRecord(value)) throw new Error('Analysis JSON must be an object.')
  if (!isRecord(value.schema)
    || value.schema.name !== 'tennis-coach-analysis'
    || value.schema.version !== 1) {
    throw new Error('Unsupported analysis schema. Expected tennis-coach-analysis version 1.')
  }
  if (!isRecord(value.data_quality)
    || !isStringArray(value.data_quality.warnings)) {
    throw new Error('Analysis JSON must include data_quality.warnings.')
  }
  if (!isRecord(value.analysis_capabilities)
    || !isStringArray(value.analysis_capabilities.supported)
    || !isStringArray(value.analysis_capabilities.unsupported)) {
    throw new Error('Analysis JSON must include supported and unsupported capabilities.')
  }
  if (!isRecord(value.players) || !Array.isArray(value.segments)) {
    throw new Error('Analysis JSON must include players and segments.')
  }
  const indexes = new Set<number>()
  for (const segment of value.segments) {
    if (!isRecord(segment)
      || !isFiniteNumber(segment.index)
      || !isFiniteNumber(segment.start)
      || !isFiniteNumber(segment.end)
      || segment.end <= segment.start) {
      throw new Error('Each segment must have a finite index and an increasing start/end range.')
    }
    if (indexes.has(segment.index)) throw new Error('Segment indexes must be unique.')
    indexes.add(segment.index)
  }
  return value as unknown as TennisAnalysis
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string')
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}
