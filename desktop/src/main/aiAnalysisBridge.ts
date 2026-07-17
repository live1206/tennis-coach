import { spawn, type ChildProcess } from 'node:child_process'
import { app, ipcMain } from 'electron'
import fs from 'node:fs'
import { createHash, randomUUID } from 'node:crypto'
import path from 'node:path'
import { validateTennisAnalysis } from '../shared/analysis'
import { processTreeSpawnOptions, terminateProcessTree } from './processControl'
import { resolveProjectPython } from './pythonRuntime'

let localAnalysisProcess: ChildProcess | null = null
let localAnalysisCancelled = false
let cloudAnalysisAbortController: AbortController | null = null

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
  const python = resolveProjectPython(getProjectRoot())
  return { cmd: python, args: ['-m', 'video_extraction.local_analysis'] }
}

function loadCanonicalAnalysis(analysisPath: string) {
  if (!path.isAbsolute(analysisPath) || path.extname(analysisPath).toLowerCase() !== '.json') {
    throw new Error('Select a completed video analysis.')
  }
  const parsed: unknown = JSON.parse(fs.readFileSync(analysisPath, 'utf-8'))
  return validateTennisAnalysis(parsed)
}

function validateAnalysisRequest(
  resolveAnalysisPath: (videoPath: string) => string,
  videoPath: string,
  evidenceId: string,
): { analysis: unknown; analysisPath: string } {
  const analysisPath = resolveAnalysisPath(videoPath)
  const analysis = loadCanonicalAnalysis(analysisPath)
  const currentEvidenceId = createHash('sha256')
    .update(JSON.stringify(analysis))
    .digest('hex')
  if (currentEvidenceId !== evidenceId) {
    throw new Error('The extracted video evidence changed. Reopen AI Analysis and try again.')
  }
  return { analysis, analysisPath }
}

function parseCloudAssistantOutput(payload: unknown): string {
  const message = (payload as Record<string, unknown>)?.choices as unknown[]
  if (!Array.isArray(message) || message.length === 0) {
    throw new Error('Cloud AI response did not contain choices.')
  }
  const first = message[0] as Record<string, unknown>
  const content = (first.message as Record<string, unknown>)?.content
  if (typeof content === 'string') return content.trim()
  if (Array.isArray(content)) {
    const combined = content
      .map((item) => {
        if (typeof item === 'string') return item
        if (item && typeof item === 'object') {
          const text = (item as Record<string, unknown>).text
          if (typeof text === 'string') return text
        }
        return ''
      })
      .join('')
      .trim()
    if (combined) return combined
  }
  throw new Error('Cloud AI response did not include textual content.')
}

async function runCloudAIAnalysis(
  analysis: unknown,
  question: string,
  modelAlias: string,
): Promise<{ output?: string; error?: string }> {
  const endpoint = process.env.TENNIS_COACH_CLOUD_API_BASE?.trim()
  const apiKey = process.env.TENNIS_COACH_CLOUD_API_KEY?.trim()
  if (!endpoint || !apiKey) {
    return {
      error: [
        'Cloud AI is not configured.',
        'Set TENNIS_COACH_CLOUD_API_BASE and TENNIS_COACH_CLOUD_API_KEY, then restart the app.',
      ].join('\n'),
    }
  }
  const apiPath = process.env.TENNIS_COACH_CLOUD_API_PATH?.trim() || '/chat/completions'
  const endpointWithSlash = endpoint.endsWith('/') ? endpoint : `${endpoint}/`
  const requestUrl = new URL(apiPath.replace(/^\/+/, ''), endpointWithSlash).toString()

  const controller = new AbortController()
  cloudAnalysisAbortController = controller
  try {
    const response = await fetch(requestUrl, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: modelAlias,
        temperature: 0,
        messages: [
          {
            role: 'system',
            content: 'You are a tennis coach assistant. Use only the supplied analysis JSON as evidence and do not invent unsupported claims.',
          },
          {
            role: 'user',
            content: [
              `Question:\n${question}`,
              '',
              'analysis.json evidence:',
              JSON.stringify(analysis),
            ].join('\n'),
          },
        ],
      }),
      signal: controller.signal,
    })
    const raw = await response.text()
    if (!response.ok) {
      const snippet = raw.slice(0, 2000)
      return { error: `Cloud AI request failed (${response.status}): ${snippet}` }
    }
    let payload: unknown
    try {
      payload = JSON.parse(raw)
    } catch {
      return { error: `Cloud AI returned non-JSON response: ${raw.slice(0, 1000)}` }
    }
    return { output: parseCloudAssistantOutput(payload) }
  } catch (error) {
    if (controller.signal.aborted) {
      return { error: 'Cloud AI analysis was cancelled.' }
    }
    return { error: error instanceof Error ? error.message : 'Cloud AI analysis failed.' }
  } finally {
    if (cloudAnalysisAbortController === controller) {
      cloudAnalysisAbortController = null
    }
  }
}

export function setupAIAnalysisBridge(
  resolveAnalysisPath: (videoPath: string) => string,
) {
  ipcMain.handle('cancel-local-ai-analysis', cancelLocalAIAnalysis)
  ipcMain.handle('cancel-ai-analysis', cancelAnyAIAnalysis)

  ipcMain.handle(
    'run-local-ai-analysis',
    async (
      _event,
      videoPath: string,
      evidenceId: string,
      question: string,
      modelAlias: string,
    ) => {
      if (localAnalysisProcess || cloudAnalysisAbortController) {
        return { error: 'An AI analysis is already running.' }
      }
      const cleanQuestion = question.trim()
      if (!cleanQuestion || cleanQuestion.length > 2000) {
        return { error: 'Question must contain between 1 and 2000 characters.' }
      }
      if (!/^[A-Za-z0-9._-]+$/.test(modelAlias)) {
        return { error: 'Model alias contains unsupported characters.' }
      }
      let snapshotPath: string
      try {
        const validated = validateAnalysisRequest(resolveAnalysisPath, videoPath, evidenceId)
        const analysis = validated.analysis
        snapshotPath = path.join(app.getPath('temp'), `tennis-coach-ai-${randomUUID()}.json`)
        fs.writeFileSync(snapshotPath, JSON.stringify(analysis), { encoding: 'utf-8', mode: 0o600 })
      } catch (error) {
        return {
          error: error instanceof Error
            ? error.message
            : 'The extracted video evidence is no longer available.',
        }
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

  ipcMain.handle(
    'run-cloud-ai-analysis',
    async (
      _event,
      videoPath: string,
      evidenceId: string,
      question: string,
      modelAlias: string,
    ) => {
      if (localAnalysisProcess || cloudAnalysisAbortController) {
        return { error: 'An AI analysis is already running.' }
      }
      const cleanQuestion = question.trim()
      if (!cleanQuestion || cleanQuestion.length > 2000) {
        return { error: 'Question must contain between 1 and 2000 characters.' }
      }
      if (!/^[A-Za-z0-9._-]+$/.test(modelAlias)) {
        return { error: 'Model alias contains unsupported characters.' }
      }
      try {
        const { analysis } = validateAnalysisRequest(resolveAnalysisPath, videoPath, evidenceId)
        return await runCloudAIAnalysis(analysis, cleanQuestion, modelAlias)
      } catch (error) {
        return {
          error: error instanceof Error
            ? error.message
            : 'The extracted video evidence is no longer available.',
        }
      }
    },
  )
}

export async function cancelLocalAIAnalysis(): Promise<void> {
  localAnalysisCancelled = true
  if (localAnalysisProcess) await terminateProcessTree(localAnalysisProcess)
}

export async function cancelAnyAIAnalysis(): Promise<void> {
  await cancelLocalAIAnalysis()
  if (cloudAnalysisAbortController) {
    cloudAnalysisAbortController.abort()
    cloudAnalysisAbortController = null
  }
}
