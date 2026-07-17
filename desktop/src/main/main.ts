import { app, BrowserWindow, dialog, ipcMain, net, protocol } from 'electron'
import path from 'node:path'
import fs from 'node:fs'
import { createHash } from 'node:crypto'
import { pathToFileURL } from 'node:url'
import Store from 'electron-store'
import {
  cancelPythonAnalysis,
  getAnalysisReportPath,
  loadCachedAnalysisReport,
  setupPythonBridge,
} from './pythonBridge'
import { cancelFfmpegExport, setupFfmpegBridge } from './ffmpegBridge'
import { cancelAnyAIAnalysis, setupAIAnalysisBridge } from './aiAnalysisBridge'
import { validateTennisAnalysis } from '../shared/analysis'

const store = new Store<{ recentProjects: string[] }>({
  defaults: { recentProjects: [] },
})

process.env.DIST_ELECTRON = path.join(__dirname, '..')
process.env.DIST = path.join(process.env.DIST_ELECTRON, '../dist')
process.env.VITE_PUBLIC = app.isPackaged
  ? process.env.DIST
  : path.join(process.env.DIST_ELECTRON, '../../public')

let win: BrowserWindow | null = null
const approvedVideoPaths = new Set<string>()
const VITE_DEV_SERVER_URL = app.isPackaged ? undefined : process.env.VITE_DEV_SERVER_URL

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'tennis-media',
    privileges: { standard: true, secure: true, stream: true, supportFetchAPI: true },
  },
])

if (process.platform === 'win32') {
  app.setAppUserModelId('com.tenniscoach.app')
}

function getWindowIconPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'icon.ico')
  }
  return path.resolve(path.join(__dirname, '../../build/icon.ico'))
}

function createWindow() {
  const productionRendererUrl = pathToFileURL(path.join(process.env.DIST!, 'index.html')).toString()
  win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 960,
    minHeight: 600,
    icon: getWindowIconPath(),
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, '../preload/preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: true,
    },
  })

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL)
  } else {
    win.loadURL(productionRendererUrl)
  }
  win.webContents.on('will-navigate', (event, url) => {
    const allowed = VITE_DEV_SERVER_URL
      ? new URL(url).origin === new URL(VITE_DEV_SERVER_URL).origin
      : url === productionRendererUrl
    if (!allowed) event.preventDefault()
  })
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
}

ipcMain.handle('open-file-dialog', async (event) => {
  const parentWin = BrowserWindow.fromWebContents(event.sender)
  const result = await dialog.showOpenDialog(parentWin!, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Video', extensions: ['mp4', 'mkv', 'avi', 'mov'] },
    ],
  })
  if (result.canceled || result.filePaths.length === 0) return null
  const filePaths = result.filePaths
  filePaths.forEach((filePath) => approvedVideoPaths.add(path.resolve(filePath)))

  const recent = store.get('recentProjects')
  const updated = [
    ...filePaths,
    ...recent.filter((p) => !filePaths.includes(p)),
  ].slice(0, 10)
  store.set('recentProjects', updated)

  return filePaths
})

ipcMain.handle('approve-dropped-video-paths', (_event, candidatePaths: unknown) => {
  if (!Array.isArray(candidatePaths)) throw new Error('Video paths must be an array.')
  const approved = candidatePaths.map((candidate) => {
    if (typeof candidate !== 'string') throw new Error('Video path must be a string.')
    const resolved = path.resolve(candidate)
    const extension = path.extname(resolved).toLowerCase()
    if (!['.mp4', '.mkv', '.avi', '.mov'].includes(extension)
      || !fs.statSync(resolved).isFile()) {
      throw new Error(`Unsupported video file: ${candidate}`)
    }
    approvedVideoPaths.add(resolved)
    return resolved
  })
  return approved
})

function loadApprovedVideoAnalysis(videoPath: string) {
  const resolvedVideoPath = path.resolve(videoPath)
  if (!approvedVideoPaths.has(resolvedVideoPath)) {
    throw new Error('Select the video through the application first.')
  }
  const parsed = loadCachedAnalysisReport(resolvedVideoPath)
  if (!parsed) throw new Error('Run video analysis before opening AI Analysis.')
  const analysis = validateTennisAnalysis(parsed)
  const evidenceId = createHash('sha256')
    .update(JSON.stringify(analysis))
    .digest('hex')
  return { videoPath: resolvedVideoPath, evidenceId, analysis }
}

ipcMain.handle('load-video-analysis', (_event, videoPath: string) => {
  return loadApprovedVideoAnalysis(videoPath)
})

ipcMain.handle('get-recent-projects', () => {
  const recent = store.get('recentProjects')
  recent.forEach((filePath) => approvedVideoPaths.add(path.resolve(filePath)))
  return recent
})

ipcMain.handle('get-app-version', () => {
  return app.getVersion()
})

ipcMain.handle('check-resources', () => {
  if (!app.isPackaged) return { ok: true, missing: [] }
  const checks = [
    { label: 'engine (TennisCoachAnalysis.exe)', path: path.join(process.resourcesPath, 'engine', 'TennisCoachAnalysis', 'TennisCoachAnalysis.exe') },
    { label: 'local AI (TennisCoachLocalAnalysis.exe)', path: path.join(process.resourcesPath, 'engine', 'TennisCoachLocalAnalysis', 'TennisCoachLocalAnalysis.exe') },
    { label: 'ffmpeg (ffmpeg.exe)', path: path.join(process.resourcesPath, 'engine', 'ffmpeg', 'ffmpeg.exe') },
    { label: 'YOLOX ball model (yolox_nano.onnx)', path: path.join(process.resourcesPath, 'engine', 'video_extraction', 'vision', 'models', 'yolox_nano.onnx') },
    { label: 'MediaPipe pose model (pose_landmarker_heavy.task)', path: path.join(process.resourcesPath, 'engine', 'video_extraction', 'vision', 'models', 'pose_landmarker_heavy.task') },
  ]
  const missing = checks.filter((c) => !fs.existsSync(c.path)).map((c) => c.label)
  return { ok: missing.length === 0, missing }
})

app.whenReady().then(() => {
  protocol.handle('tennis-media', (request) => {
    const requestedPath = path.resolve(
      decodeURIComponent(new URL(request.url).pathname.slice(1)),
    )
    if (!approvedVideoPaths.has(requestedPath)) {
      return new Response('Forbidden', { status: 403 })
    }
    return net.fetch(pathToFileURL(requestedPath).toString())
  })
  setupPythonBridge((videoPath) => approvedVideoPaths.has(path.resolve(videoPath)))
  setupFfmpegBridge((videoPath) => approvedVideoPaths.has(path.resolve(videoPath)))
  setupAIAnalysisBridge((videoPath) => {
    const loaded = loadApprovedVideoAnalysis(videoPath)
    return getAnalysisReportPath(loaded.videoPath)
  })
  createWindow()
})

app.on('window-all-closed', () => {
  win = null
  app.quit()
})

let shutdownStarted = false
app.on('before-quit', (event) => {
  if (shutdownStarted) return
  event.preventDefault()
  shutdownStarted = true
  Promise.allSettled([
    cancelPythonAnalysis(),
    cancelAnyAIAnalysis(),
    cancelFfmpegExport(),
  ]).finally(() => app.exit(0))
})
