import { spawn, type ChildProcess, type SpawnOptions } from 'node:child_process'

export const processTreeSpawnOptions: Pick<SpawnOptions, 'detached' | 'windowsHide'> = {
  detached: process.platform !== 'win32',
  windowsHide: true,
}

export async function terminateProcessTree(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null || !child.pid) return
  const exited = new Promise<void>((resolve) => {
    child.once('close', () => resolve())
    child.once('error', () => resolve())
  })
  terminate(child, false)
  if (await waitFor(exited, 3000)) return
  terminate(child, true)
  await waitFor(exited, 2000)
}

function terminate(child: ChildProcess, force: boolean) {
  if (!child.pid) return
  if (process.platform === 'win32') {
    spawn('taskkill', ['/pid', String(child.pid), '/T', ...(force ? ['/F'] : [])], {
      stdio: 'ignore',
      windowsHide: true,
    })
    return
  }
  try {
    process.kill(-child.pid, force ? 'SIGKILL' : 'SIGTERM')
  } catch {
    child.kill(force ? 'SIGKILL' : 'SIGTERM')
  }
}

async function waitFor(promise: Promise<void>, timeoutMs: number): Promise<boolean> {
  let timeout: NodeJS.Timeout | undefined
  const timedOut = new Promise<false>((resolve) => {
    timeout = setTimeout(() => resolve(false), timeoutMs)
    timeout.unref()
  })
  const result = await Promise.race([promise.then(() => true), timedOut])
  if (timeout) clearTimeout(timeout)
  return result
}
