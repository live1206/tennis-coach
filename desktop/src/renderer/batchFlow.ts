import type { RallyPlayers, RallySegment, VideoRecord } from './state/AppState'
import type { TennisAnalysis } from '../shared/analysis'
export type { RallySegment, VideoRecord } from './state/AppState'

const AUTO_INCLUDE_THRESHOLD = 1.7

export interface ExportClip {
  videoPath: string
  start: number
  end: number
}

export function getVideoDisplayName(path: string): string {
  return path.split(/[\\/]/).pop() || path
}

export function createVideoRecords(paths: string[]): VideoRecord[] {
  return paths.map((path, order) => ({
    id: `video-${order + 1}`,
    path,
    displayName: getVideoDisplayName(path),
    order,
    status: 'pending',
    errorMessage: null,
    currentStep: null,
    duration: 0,
    rallyCount: 0,
  }))
}

export function createRalliesForVideo(video: VideoRecord, analysis: TennisAnalysis): RallySegment[] {
  return analysis.segments.map((rawSegment) => {
    const segment = rawSegment as Record<string, unknown>
    const motion = asRecord(segment.motion)
    const audio = asRecord(segment.audio)
    const ball = asRecord(segment.ball)
    const score = numberValue(motion.player_motion_max) * 100
    return {
    id: `${video.id}-rally-${segment.index}`,
    videoId: video.id,
    sourceIndex: numberValue(segment.index),
    index: numberValue(segment.index),
    start: numberValue(segment.start),
    end: numberValue(segment.end),
    score,
    features: {
      hit_count: numberValue(audio.hit_count),
      ball_visible_ratio: numberValue(ball.visible_ratio),
    },
    players: mapPlayers(asRecord(segment.players)),
    included: score > AUTO_INCLUDE_THRESHOLD,
  }})
}

function mapPlayers(players: Record<string, unknown>): RallyPlayers | undefined {
  const player1 = asRecord(players.player_1)
  const player2 = asRecord(players.player_2)
  if (Object.keys(player1).length === 0 || Object.keys(player2).length === 0) return undefined
  return {
    player_1: {
      detected: Boolean(player1.detected),
      side: player1.side === 'near' || player1.side === 'far' ? player1.side : undefined,
      detection_confidence: optionalNumber(player1.detection_confidence),
      identity_confidence: optionalNumber(player1.identity_confidence),
      movement_distance: optionalNumber(player1.image_movement_distance_normalized),
      sample_count: optionalNumber(player1.trajectory_samples),
    },
    player_2: {
      detected: Boolean(player2.detected),
      side: player2.side === 'near' || player2.side === 'far' ? player2.side : undefined,
      detection_confidence: optionalNumber(player2.detection_confidence),
      identity_confidence: optionalNumber(player2.identity_confidence),
      movement_distance: optionalNumber(player2.image_movement_distance_normalized),
      sample_count: optionalNumber(player2.trajectory_samples),
    },
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function numberValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

export function getSortedRallies(rallies: RallySegment[], videos: VideoRecord[]): RallySegment[] {
  const orderByVideo = new Map(videos.map((video) => [video.id, video.order]))
  return rallies.slice().sort((a, b) => {
    const videoDelta = (orderByVideo.get(a.videoId) ?? Number.MAX_SAFE_INTEGER) - (orderByVideo.get(b.videoId) ?? Number.MAX_SAFE_INTEGER)
    if (videoDelta !== 0) return videoDelta
    return a.start - b.start
  })
}

export function getRalliesForVideo(rallies: RallySegment[], videoId: string): RallySegment[] {
  return rallies.filter((rally) => rally.videoId === videoId)
}

export function getExportClips(rallies: RallySegment[], videos: VideoRecord[]): ExportClip[] {
  const pathByDoneVideo = new Map(videos.filter((video) => video.status === 'done').map((video) => [video.id, video.path]))
  return getSortedRallies(rallies.filter((rally) => rally.included), videos)
    .map((rally) => {
      const videoPath = pathByDoneVideo.get(rally.videoId)
      if (!videoPath) return null
      return {
        videoPath,
        start: rally.startAdjusted ?? rally.start,
        end: rally.endAdjusted ?? rally.end,
      }
    })
    .filter((clip): clip is ExportClip => clip !== null && clip.end > clip.start)
}
