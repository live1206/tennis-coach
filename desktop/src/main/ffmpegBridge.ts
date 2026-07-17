import { spawn, ChildProcess } from 'node:child_process'
import { app, ipcMain, dialog, BrowserWindow, shell } from 'electron'
import { unlink } from 'node:fs/promises'
import path from 'node:path'
import { processTreeSpawnOptions, terminateProcessTree } from './processControl'

interface ActiveExport {
  process: ChildProcess
  outputPath: string
  cancelled: boolean
}

let activeExport: ActiveExport | null = null
const completedExportPaths = new Set<string>()

function getFfmpegPath(): string {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'engine', 'ffmpeg', 'ffmpeg.exe')
  }
  return 'ffmpeg'
}

interface ExportClip {
  videoPath: string
  start: number
  end: number
}

export function setupFfmpegBridge(isApprovedVideoPath: (videoPath: string) => boolean) {
  ipcMain.handle('cancel-export', cancelFfmpegExport)

  ipcMain.handle('open-export-path', async (_event, targetPath: unknown) => {
    if (typeof targetPath !== 'string') throw new Error('Export path must be a string.')
    const resolved = path.resolve(targetPath)
    if (!completedExportPaths.has(resolved)) {
      throw new Error('Only a completed highlight export can be opened.')
    }
    return shell.openPath(resolved)
  })

  ipcMain.handle('export-highlights', async (event, clips: unknown) => {
    if (activeExport) return { error: 'A highlight export is already running' }
    if (!Array.isArray(clips)) return { error: 'Highlight clips must be an array' }
    if (clips.some((clip): clip is ExportClip => (
      !clip
      || typeof clip !== 'object'
      || typeof clip.videoPath !== 'string'
      || !Number.isFinite(clip.start)
      || !Number.isFinite(clip.end)
    ))) {
      return { error: 'Invalid highlight clip data' }
    }
    if (clips.some((clip) => !isApprovedVideoPath(clip.videoPath))) {
      return { error: 'Select every source video through the application first.' }
    }

    const win = BrowserWindow.fromWebContents(event.sender)
    const sorted = [...clips].filter((clip) => clip.end > clip.start)
    const firstClip = sorted[0]
    if (!firstClip) return { error: 'No clips selected for export' }

    const result = await dialog.showSaveDialog({
      defaultPath: path.join(
        path.dirname(firstClip.videoPath),
        `${path.basename(firstClip.videoPath, path.extname(firstClip.videoPath))}_highlights.mp4`,
      ),
      filters: [{ name: 'MP4', extensions: ['mp4'] }],
    })

    if (result.canceled || !result.filePath) return { cancelled: true }
    const outputPath = path.extname(result.filePath).toLowerCase() === '.mp4'
      ? path.resolve(result.filePath)
      : path.resolve(`${result.filePath}.mp4`)

    const inputs: string[] = []
    const filterChains: string[] = []
    const concatParts: string[] = []
    for (let i = 0; i < sorted.length; i++) {
      const clip = sorted[i]
      inputs.push('-ss', String(clip.start), '-t', String(clip.end - clip.start), '-i', clip.videoPath)
      filterChains.push(
        `[${i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v${i}]`,
        `[${i}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a${i}]`,
      )
      concatParts.push(`[v${i}][a${i}]`)
    }

    const filterStr = `${filterChains.join(';')};${concatParts.join('')}concat=n=${sorted.length}:v=1:a=1[outv][outa]`

    const ffmpeg = getFfmpegPath()
    const cmd = [ffmpeg, '-y', ...inputs, '-filter_complex', filterStr, '-map', '[outv]', '-map', '[outa]', outputPath]

    return new Promise<{ error?: string; cancelled?: boolean; outputPath?: string }>((resolve) => {
      const proc = spawn(cmd[0], cmd.slice(1), {
        stdio: ['ignore', 'pipe', 'pipe'],
        ...processTreeSpawnOptions,
      })
      const currentExport: ActiveExport = { process: proc, outputPath, cancelled: false }
      activeExport = currentExport
      let settled = false
      const settle = (result: { error?: string; cancelled?: boolean; outputPath?: string }) => {
        if (settled) return
        settled = true
        resolve(result)
      }

      proc.stderr?.on('data', (data: Buffer) => {
        const line = data.toString()
        const match = line.match(/time=(\d+):(\d+):(\d+\.\d+)/)
        if (match) {
          const secs = parseInt(match[1]) * 3600 + parseInt(match[2]) * 60 + parseFloat(match[3])
          win?.webContents.send('export-progress', { time: secs })
        }
      })

      proc.on('close', (code, signal) => {
        if (activeExport === currentExport) activeExport = null
        if (currentExport.cancelled || signal === 'SIGTERM' || signal === 'SIGKILL') {
          unlink(outputPath).catch(() => {})
          settle({ cancelled: true })
        } else if (code === 0) {
          completedExportPaths.add(outputPath)
          settle({ outputPath })
        } else {
          settle({ error: `ffmpeg exited with code ${code}` })
        }
      })

      proc.on('error', (err) => {
        if (activeExport === currentExport) activeExport = null
        if (currentExport.cancelled) {
          unlink(outputPath).catch(() => {})
          settle({ cancelled: true })
        } else {
          settle({ error: err.message })
        }
      })
    })
  })
}

export async function cancelFfmpegExport(): Promise<void> {
  const currentExport = activeExport
  if (!currentExport) return
  currentExport.cancelled = true
  await terminateProcessTree(currentExport.process)
}
