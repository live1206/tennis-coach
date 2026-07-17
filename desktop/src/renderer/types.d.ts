export {}

import type { LoadedAnalysis, TennisAnalysis } from '../shared/analysis'

interface ProgressEvent {
  type: 'step' | 'step_done' | 'complete' | 'error' | 'progress' | 'stderr'
  step?: number
  total?: number
  label?: string
  elapsed?: number
  detail?: Record<string, number>
  report_path?: string
  segment_count?: number
  message?: string
  current?: number
  sub_total?: number
}

interface Segment {
  analysis_version?: number
  player_identity_status?: 'complete' | 'skipped_court_detection' | 'disabled'
  index: number
  start: number
  end: number
  score: number
  features: Record<string, number>
  players?: {
    player_1: PlayerIdentity
    player_2: PlayerIdentity
  }
}

interface PlayerIdentity {
  detected: boolean
  side?: 'near' | 'far'
  detection_confidence?: number
  identity_confidence?: number
  movement_distance?: number
  sample_count?: number
  mean_position?: [number, number]
}

declare global {
  interface Window {
    api: {
      openFileDialog: () => Promise<string[] | null>
      resolveDroppedVideoPaths: (files: File[]) => Promise<string[]>
      getVideoUrl: (videoPath: string) => string
      loadVideoAnalysis: (videoPath: string) => Promise<LoadedAnalysis>
      runLocalAIAnalysis: (
        videoPath: string,
        evidenceId: string,
        question: string,
        modelAlias: string,
      ) => Promise<{ output?: string; error?: string }>
      runCloudAIAnalysis: (
        videoPath: string,
        evidenceId: string,
        question: string,
        modelAlias: string,
      ) => Promise<{ output?: string; error?: string }>
      cancelLocalAIAnalysis: () => Promise<void>
      cancelAIAnalysis: () => Promise<void>
      getRecentProjects: () => Promise<string[]>
      getAppVersion: () => Promise<string>
      checkResources: () => Promise<{ ok: boolean; missing: string[] }>
      openExportPath: (targetPath: string) => Promise<string>
      runAnalysis: (videoPath: string) => Promise<{ error?: string }>
      cancelAnalysis: () => Promise<void>
      cancelExport: () => Promise<void>
      loadReport: (videoPath: string) => Promise<TennisAnalysis | null>
      exportHighlights: (clips: { videoPath: string; start: number; end: number }[]) =>
        Promise<{ error?: string; cancelled?: boolean; outputPath?: string }>
      onAnalysisProgress: (callback: (event: ProgressEvent) => void) => () => void
      onExportProgress: (callback: (event: { time: number }) => void) => () => void
    }
  }
}
