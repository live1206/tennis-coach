import { spawn, ChildProcess } from 'node:child_process'
import { createHash } from 'node:crypto'
import { app, ipcMain, BrowserWindow } from 'electron'
import path from 'node:path'
import fs from 'node:fs'
import { processTreeSpawnOptions, terminateProcessTree } from './processControl'
import {
  getBallModelSetupHint,
  formatPythonDependencyError,
  resolveBallModelPath,
  resolvePlayerHandedness,
  resolvePlayerHandednessArgs,
  resolvePoseModelPath,
  resolveProjectPython,
} from './pythonRuntime'

let analysisProcess: ChildProcess | null = null
let analysisCancellationPromise: Promise<void> | null = null
const EXTRACTION_CONFIG_VERSION = 1
const DEFAULT_POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'

function clearAnalysisProcess(child: ChildProcess) {
  if (analysisProcess === child) {
    analysisProcess = null
    analysisCancellationPromise = null
  }
}

function getEngineCommand(): { cmd: string; args: string[] } {
  if (app.isPackaged) {
    const enginePath = path.join(process.resourcesPath, 'engine', 'TennisCoachAnalysis', 'TennisCoachAnalysis.exe')
    return { cmd: enginePath, args: [] }
  }
  const python = resolveProjectPython(getProjectRoot())
  return { cmd: python, args: ['-m', 'video_extraction.cli'] }
}

function getProjectRoot(): string {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'engine')
  }
  return path.resolve(path.join(__dirname, '../../..'))
}

function getOutputDir(videoPath: string): string {
  const sourceName = path.basename(videoPath).replace(/[^A-Za-z0-9._-]/g, '_')
  const sourceHash = createHash('sha256')
    .update(path.resolve(videoPath))
    .digest('hex')
    .slice(0, 12)
  return path.join(path.dirname(videoPath), `output_${sourceName}_${sourceHash}`)
}

export function getAnalysisReportPath(videoPath: string): string {
  return path.join(getOutputDir(videoPath), 'analysis.json')
}

function getAnalysisErrorLogPath(videoPath: string): string {
  return path.join(getOutputDir(videoPath), 'analysis-error.log')
}

function writeAnalysisErrorLog(
  videoPath: string,
  commandLine: string,
  cwd: string,
  stdout: string,
  stderr: string,
  errorDetail: string,
): string | null {
  const logPath = getAnalysisErrorLogPath(videoPath)
  const body = [
    `timestamp=${new Date().toISOString()}`,
    `video=${videoPath}`,
    `cwd=${cwd}`,
    `command=${commandLine}`,
    '',
    '=== ERROR ===',
    errorDetail || '(none)',
    '',
    '=== STDERR ===',
    stderr || '(none)',
    '',
    '=== STDOUT ===',
    stdout || '(none)',
    '',
  ].join('\n')
  try {
    fs.writeFileSync(logPath, body, 'utf-8')
    return logPath
  } catch {
    return null
  }
}

function appendLogPath(error: string, logPath: string | null): string {
  if (!logPath) return error
  return `${error}\n\nFull extraction log:\n${logPath}`
}

function getExtractionConfig(projectRoot: string) {
  return {
    version: EXTRACTION_CONFIG_VERSION,
    poseModelPath: resolvePoseModelPath(projectRoot),
    ballModelPath: resolveBallModelPath(projectRoot),
    handedness: resolvePlayerHandedness(),
  }
}

function getPoseModelTargetPath(projectRoot: string): string {
  const configured = process.env.TENNIS_COACH_POSE_MODEL_PATH?.trim()
  if (configured) {
    return path.resolve(configured)
  }
  if (app.isPackaged) {
    return path.join(app.getPath('userData'), 'models', 'pose_landmarker_heavy.task')
  }
  return path.join(projectRoot, 'models', 'pose_landmarker_heavy.task')
}

async function downloadPoseModel(targetPath: string): Promise<string> {
  const modelUrl = process.env.TENNIS_COACH_POSE_MODEL_URL?.trim() || DEFAULT_POSE_MODEL_URL
  const response = await fetch(modelUrl)
  if (!response.ok) {
    throw new Error(`Pose model download failed (${response.status}) from ${modelUrl}`)
  }
  const bytes = Buffer.from(await response.arrayBuffer())
  fs.mkdirSync(path.dirname(targetPath), { recursive: true })
  fs.writeFileSync(targetPath, bytes)
  return targetPath
}

async function ensurePoseModelPath(projectRoot: string): Promise<string> {
  const existing = resolvePoseModelPath(projectRoot)
  if (existing) return existing
  const targetPath = getPoseModelTargetPath(projectRoot)
  if (fs.existsSync(targetPath)) return targetPath
  return downloadPoseModel(targetPath)
}

function getSourceMetadata(
  videoPath: string,
  extractionConfig: ReturnType<typeof getExtractionConfig> = getExtractionConfig(getProjectRoot()),
) {
  const stats = fs.statSync(videoPath)
  return {
    path: path.resolve(videoPath),
    size: stats.size,
    mtimeMs: stats.mtimeMs,
    extractionConfig,
  }
}

function sourceMetadataMatches(
  before: ReturnType<typeof getSourceMetadata>,
  after: ReturnType<typeof getSourceMetadata>,
): boolean {
  const beforeConfig = JSON.stringify(before.extractionConfig ?? null)
  const afterConfig = JSON.stringify(after.extractionConfig ?? null)
  return before.path === after.path
    && before.size === after.size
    && before.mtimeMs === after.mtimeMs
    && beforeConfig === afterConfig
}

export function loadCachedAnalysisReport(videoPath: string): unknown | null {
  const outputDir = getOutputDir(videoPath)
  const reportPath = getAnalysisReportPath(videoPath)
  const sourcePath = path.join(outputDir, 'source.json')
  if (!fs.existsSync(reportPath) || !fs.existsSync(sourcePath)) return null

  const recorded = JSON.parse(fs.readFileSync(sourcePath, 'utf-8'))
  const current = getSourceMetadata(videoPath)
  if (!sourceMetadataMatches(recorded, current)) return null
  return JSON.parse(fs.readFileSync(reportPath, 'utf-8'))
}

export function setupPythonBridge(isApprovedVideoPath: (videoPath: string) => boolean) {
  ipcMain.handle('run-analysis', async (event, videoPath: string) => {
    if (analysisProcess) {
      return { error: 'Analysis already running' }
    }
    if (!isApprovedVideoPath(videoPath)) {
      return { error: 'Select the video through the application first.' }
    }

    const { cmd, args } = getEngineCommand()
    const cwd = getProjectRoot()
    let poseModelPath: string
    try {
      poseModelPath = await ensurePoseModelPath(cwd)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to prepare pose model.'
      return {
        error: [
          'Could not prepare the MediaPipe Pose Landmarker model automatically.',
          message,
          'Set TENNIS_COACH_POSE_MODEL_PATH manually if needed.',
        ].join('\n'),
      }
    }
    const yoloxModelPath = resolveBallModelPath(cwd)
    if (!yoloxModelPath) {
      return { error: getBallModelSetupHint(cwd) }
    }
    const outputDir = getOutputDir(videoPath)
    const sourcePath = path.join(outputDir, 'source.json')
    const errorLogPath = getAnalysisErrorLogPath(videoPath)
    let sourceMetadata: ReturnType<typeof getSourceMetadata>
    try {
      sourceMetadata = getSourceMetadata(videoPath)
      fs.mkdirSync(outputDir, { recursive: true })
      fs.rmSync(sourcePath, { force: true })
      fs.rmSync(errorLogPath, { force: true })
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'Could not prepare the analysis output.' }
    }
    const fullArgs = [
      ...args,
      videoPath,
      '--pose-model-path',
      poseModelPath,
      '--model-path',
      yoloxModelPath,
      '--ball-detector',
      'yolox',
      '--ball-model-path',
      yoloxModelPath,
      ...resolvePlayerHandednessArgs(),
      '--output',
      getAnalysisReportPath(videoPath),
      '--internal-output-dir',
      path.join(outputDir, 'internal'),
    ]
    const commandLine = `${cmd} ${fullArgs.join(' ')}`

    const env = { ...process.env }
    if (app.isPackaged) {
      const ffmpegDir = path.join(process.resourcesPath, 'engine', 'ffmpeg')
      env.PATH = ffmpegDir + path.delimiter + (env.PATH ?? '')
    }

    return new Promise<{ error?: string }>((resolve) => {
      let settled = false
      const settle = (result: { error?: string }) => {
        if (settled) return
        settled = true
        resolve(result)
      }

      const child = spawn(cmd, fullArgs, {
        cwd,
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
        ...processTreeSpawnOptions,
      })
      analysisProcess = child

      const win = BrowserWindow.fromWebContents(event.sender)
      win?.webContents.send('analysis-progress', {
        type: 'step',
        step: 1,
        total: 1,
        label: 'Extracting canonical tennis analysis',
      })
      let stdoutBuf = ''
      let stdoutAll = ''
      let stderrBuf = ''
      let completionError: string | null = null

      const handleStdoutLine = (line: string) => {
        const trimmed = line.trim()
        if (!trimmed) return
        try {
          const msg = JSON.parse(trimmed)
          win?.webContents.send('analysis-progress', msg)
          if (msg.type === 'error') {
            completionError = msg.traceback ?? msg.message ?? 'Analysis failed'
          }
        } catch {
          // non-JSON output, ignore
        }
      }

      child.stdout?.on('data', (data: Buffer) => {
        const text = data.toString()
        stdoutBuf += text
        stdoutAll += text
        const parts = stdoutBuf.split('\n')
        stdoutBuf = parts.pop()!
        for (const line of parts) {
          handleStdoutLine(line)
        }
      })

      child.stderr?.on('data', (data: Buffer) => {
        const text = data.toString()
        stderrBuf += text
        console.error('[python]', text)
        win?.webContents.send('analysis-progress', { type: 'stderr', message: text })
      })

      child.on('close', (code) => {
        if (stdoutBuf.trim()) {
          handleStdoutLine(stdoutBuf)
        }
        clearAnalysisProcess(child)
        if (completionError) {
          const logPath = writeAnalysisErrorLog(
            videoPath,
            commandLine,
            cwd,
            stdoutAll,
            stderrBuf,
            completionError,
          )
          settle({
            error: appendLogPath(
              formatPythonDependencyError(completionError),
              logPath,
            ),
          })
        } else if (code !== 0) {
          const detail = stderrBuf.trim()
          const failure = detail || `Process exited with code ${code}`
          const logPath = writeAnalysisErrorLog(
            videoPath,
            commandLine,
            cwd,
            stdoutAll,
            stderrBuf,
            failure,
          )
          settle({
            error: appendLogPath(
              formatPythonDependencyError(failure),
              logPath,
            ),
          })
        } else {
          try {
            const currentMetadata = getSourceMetadata(videoPath)
            if (!sourceMetadataMatches(sourceMetadata, currentMetadata)) {
              settle({ error: 'The source video changed while analysis was running. Run the analysis again.' })
              return
            }
            fs.writeFileSync(
              sourcePath,
              JSON.stringify(sourceMetadata, null, 2),
            )
          } catch (error) {
            settle({ error: error instanceof Error ? error.message : 'Failed to save source metadata.' })
            return
          }
          win?.webContents.send('analysis-progress', { type: 'complete' })
          settle({})
        }
      })

      child.on('error', (err) => {
        clearAnalysisProcess(child)
        const logPath = writeAnalysisErrorLog(
          videoPath,
          commandLine,
          cwd,
          stdoutAll,
          stderrBuf,
          err.message,
        )
        settle({
          error: appendLogPath(
            formatPythonDependencyError(err.message),
            logPath,
          ),
        })
      })
    })
  })

  ipcMain.handle('cancel-analysis', async () => {
    return cancelPythonAnalysis()
  })

  ipcMain.handle('load-report', async (_event, reportOrVideoPath: string) => {
    if (!isApprovedVideoPath(reportOrVideoPath)) {
      throw new Error('Select the video through the application first.')
    }
    return loadCachedAnalysisReport(reportOrVideoPath)
  })
}

export async function cancelPythonAnalysis(): Promise<void> {
  const processToCancel = analysisProcess
  if (!processToCancel) return
  const cancellation = analysisCancellationPromise
    ?? terminateProcessTree(processToCancel)
  analysisCancellationPromise = cancellation
  await cancellation
}
