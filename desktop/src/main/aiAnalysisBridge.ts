import { spawn, type ChildProcess } from 'node:child_process'
import { app, ipcMain } from 'electron'
import fs from 'node:fs'
import { createHash, randomUUID } from 'node:crypto'
import path from 'node:path'
import { validateTennisAnalysis } from '../shared/analysis'
import { processTreeSpawnOptions, terminateProcessTree } from './processControl'

let localAnalysisProcess: ChildProcess | null = null
let localAnalysisCancelled = false

function getProjectRoot(): string {
  return app.isPackaged ? process.resourcesPath : path.resolve(app.getAppPath(), '..')
}

function getLocalAnalysisCommand(): { cmd: string; args: string[] } {
  if (app.isPackaged) {
    return {
      cmd: path.join(
        process.resourcesPath,
        'engine',
        'TennisCoachLocalAnalysis',
        'TennisCoachLocalAnalysis.exe',
      ),
      args: [],
    }
  }
  const python = process.env.TENNIS_COACH_PYTHON
    || (process.platform === 'win32' ? 'python' : 'python3')
  return { cmd: python, args: ['-m', 'video_extraction.local_analysis'] }
}

function loadCanonicalAnalysis(analysisPath: string) {
  if (!path.isAbsolute(analysisPath) || path.extname(analysisPath).toLowerCase() !== '.json') {
    throw new Error('Select a completed video analysis.')
  }
  const parsed: unknown = JSON.parse(fs.readFileSync(analysisPath, 'utf-8'))
  return validateTennisAnalysis(parsed)
}

export function setupAIAnalysisBridge(
  resolveAnalysisPath: (videoPath: string) => string,
) {
  ipcMain.handle('cancel-local-ai-analysis', cancelLocalAIAnalysis)

  ipcMain.handle(
    'run-local-ai-analysis',
    async (
      _event,
      videoPath: string,
      evidenceId: string,
      question: string,
      modelAlias: string,
    ) => {
      if (localAnalysisProcess) return { error: 'A local AI analysis is already running.' }
      let analysisPath: string
      let snapshotPath: string
      try {
        analysisPath = resolveAnalysisPath(videoPath)
        const analysis = loadCanonicalAnalysis(analysisPath)
        const currentEvidenceId = createHash('sha256')
          .update(JSON.stringify(analysis))
          .digest('hex')
        if (currentEvidenceId !== evidenceId) {
          throw new Error('The extracted video evidence changed. Reopen AI Analysis and try again.')
        }
        snapshotPath = path.join(app.getPath('temp'), `tennis-coach-ai-${randomUUID()}.json`)
        fs.writeFileSync(snapshotPath, JSON.stringify(analysis), { encoding: 'utf-8', mode: 0o600 })
      } catch (error) {
        return {
          error: error instanceof Error
            ? error.message
            : 'The extracted video evidence is no longer available.',
        }
      }
      const cleanQuestion = question.trim()
      if (!cleanQuestion || cleanQuestion.length > 2000) {
        return { error: 'Question must contain between 1 and 2000 characters.' }
      }
      if (!/^[A-Za-z0-9._-]+$/.test(modelAlias)) {
        return { error: 'Model alias contains unsupported characters.' }
      }

      const command = getLocalAnalysisCommand()
      const args = [
        ...command.args,
        snapshotPath,
        '--question',
        cleanQuestion,
        '--model',
        modelAlias,
      ]

      return new Promise<{ output?: string; error?: string }>((resolve) => {
        let settled = false
        const settle = (result: { output?: string; error?: string }) => {
          if (settled) return
          settled = true
          resolve(result)
        }
        const child = spawn(command.cmd, args, {
          cwd: getProjectRoot(),
          env: process.env,
          stdio: ['ignore', 'pipe', 'pipe'],
          ...processTreeSpawnOptions,
        })
        localAnalysisProcess = child
        localAnalysisCancelled = false
        let stdout = ''
        let stderr = ''
        const append = (current: string, data: Buffer) => (
          current + data.toString('utf-8')
        ).slice(-1_000_000)
        child.stdout?.on('data', (data: Buffer) => { stdout = append(stdout, data) })
        child.stderr?.on('data', (data: Buffer) => { stderr = append(stderr, data) })
        child.on('error', (error) => {
          if (localAnalysisProcess === child) localAnalysisProcess = null
          fs.rmSync(snapshotPath, { force: true })
          settle({ error: error.message })
        })
        child.on('close', (code) => {
          if (localAnalysisProcess === child) localAnalysisProcess = null
          fs.rmSync(snapshotPath, { force: true })
          if (localAnalysisCancelled) {
            settle({ error: 'Local AI analysis was cancelled.' })
          } else if (code === 0) {
            settle({ output: stdout.trim() })
          } else {
            settle({ error: stderr.trim() || `Local analysis exited with code ${code}.` })
          }
        })
      })
    },
  )
}

export async function cancelLocalAIAnalysis(): Promise<void> {
  localAnalysisCancelled = true
  if (localAnalysisProcess) await terminateProcessTree(localAnalysisProcess)
}
