import fs from 'node:fs'
import path from 'node:path'

export function resolveProjectPython(projectRoot: string): string {
  const configuredPython = process.env.TENNIS_COACH_PYTHON?.trim()
  if (configuredPython) {
    return configuredPython
  }
  const venvPython = process.platform === 'win32'
    ? path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(projectRoot, '.venv', 'bin', 'python')
  if (fs.existsSync(venvPython)) {
    return venvPython
  }
  return process.platform === 'win32' ? 'python' : 'python3'
}

export function formatPythonDependencyError(detail: string): string {
  if (/ModuleNotFoundError:\s*No module named ['"]cv2['"]/i.test(detail)) {
    return [
      'Missing Python dependency: cv2 (OpenCV).',
      'Install project dependencies in the environment used by Tennis Coach:',
      'python -m pip install -e .',
      'Or set TENNIS_COACH_PYTHON to a Python environment where dependencies are installed.',
    ].join('\n')
  }
  if (/'ffmpeg' not found|Install ffmpeg/i.test(detail)) {
    return [
      'Missing dependency: ffmpeg.',
      'Install ffmpeg (Ubuntu/WSL): sudo apt update && sudo apt install -y ffmpeg',
      'Or set TENNIS_COACH_FFMPEG to the ffmpeg binary path.',
      '',
      detail,
    ].join('\n')
  }
  return detail
}
