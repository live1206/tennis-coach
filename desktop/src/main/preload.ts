import { contextBridge, ipcRenderer, webUtils } from 'electron'

export interface ProgressEvent {
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

contextBridge.exposeInMainWorld('api', {
  openFileDialog: () => ipcRenderer.invoke('open-file-dialog') as Promise<string[] | null>,
  resolveDroppedVideoPaths: (files: File[]) => {
    const videoPaths = files
      .map((file) => webUtils.getPathForFile(file))
      .filter(Boolean)
    return ipcRenderer.invoke('approve-dropped-video-paths', videoPaths) as Promise<string[]>
  },
  getVideoUrl: (videoPath: string) => `tennis-media://local/${encodeURIComponent(videoPath)}`,
  loadVideoAnalysis: (videoPath: string) => ipcRenderer.invoke('load-video-analysis', videoPath),
  runLocalAIAnalysis: (
    videoPath: string,
    evidenceId: string,
    question: string,
    modelAlias: string,
  ) => ipcRenderer.invoke('run-local-ai-analysis', videoPath, evidenceId, question, modelAlias),
  runCloudAIAnalysis: (
    videoPath: string,
    evidenceId: string,
    question: string,
  ) => ipcRenderer.invoke('run-cloud-ai-analysis', videoPath, evidenceId, question),
  cancelLocalAIAnalysis: () => ipcRenderer.invoke('cancel-local-ai-analysis') as Promise<void>,
  cancelCloudAIAnalysis: () => ipcRenderer.invoke('cancel-cloud-ai-analysis') as Promise<void>,
  getRecentProjects: () => ipcRenderer.invoke('get-recent-projects'),
  getAppVersion: () => ipcRenderer.invoke('get-app-version') as Promise<string>,
  checkResources: () => ipcRenderer.invoke('check-resources') as Promise<{ ok: boolean; missing: string[] }>,
  openExportPath: (targetPath: string) =>
    ipcRenderer.invoke('open-export-path', targetPath) as Promise<string>,
  runAnalysis: (videoPath: string) => ipcRenderer.invoke('run-analysis', videoPath),
  cancelAnalysis: () => ipcRenderer.invoke('cancel-analysis'),
  cancelExport: () => ipcRenderer.invoke('cancel-export'),
  loadReport: (videoPath: string) => ipcRenderer.invoke('load-report', videoPath),
  exportHighlights: (clips: { videoPath: string; start: number; end: number }[]) =>
    ipcRenderer.invoke('export-highlights', clips),
  onAnalysisProgress: (callback: (event: ProgressEvent) => void) => {
    const handler = (_: unknown, data: ProgressEvent) => callback(data)
    ipcRenderer.on('analysis-progress', handler)
    return () => ipcRenderer.removeListener('analysis-progress', handler)
  },
  onExportProgress: (callback: (event: { time: number }) => void) => {
    const handler = (_: unknown, data: { time: number }) => callback(data)
    ipcRenderer.on('export-progress', handler)
    return () => ipcRenderer.removeListener('export-progress', handler)
  },
})
