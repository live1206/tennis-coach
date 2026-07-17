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

export function resolvePoseModelPath(projectRoot: string): string | null {
  const configuredModel = process.env.TENNIS_COACH_POSE_MODEL_PATH?.trim()
  if (configuredModel) {
    const candidate = path.resolve(configuredModel)
    return fs.existsSync(candidate) ? candidate : null
  }
  const candidates = [
    path.join(projectRoot, 'models', 'pose_landmarker_heavy.task'),
    path.join(projectRoot, 'video_extraction', 'vision', 'models', 'pose_landmarker_heavy.task'),
    path.join(projectRoot, 'third_party', 'mediapipe', 'pose_landmarker_heavy.task'),
  ]
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null
}

export function resolveBallModelPath(projectRoot: string): string | null {
  const configuredModel = process.env.TENNIS_COACH_BALL_MODEL_PATH?.trim()
  if (configuredModel) {
    const candidate = path.resolve(configuredModel)
    return fs.existsSync(candidate) ? candidate : null
  }
  const candidates = [
    path.join(projectRoot, 'video_extraction', 'vision', 'models', 'yolox_nano.onnx'),
    path.join(projectRoot, 'models', 'yolox_nano.onnx'),
  ]
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null
}

export function getPoseModelSetupHint(projectRoot: string): string {
  const suggestedPath = path.join(projectRoot, 'models', 'pose_landmarker_heavy.task')
  return [
    'Missing MediaPipe Pose Landmarker model.',
    'Set TENNIS_COACH_POSE_MODEL_PATH to pose_landmarker_heavy.task and restart the app.',
    `Example: TENNIS_COACH_POSE_MODEL_PATH=${suggestedPath}`,
  ].join('\n')
}

export function getBallModelSetupHint(projectRoot: string): string {
  const suggestedPath = path.join(projectRoot, 'video_extraction', 'vision', 'models', 'yolox_nano.onnx')
  return [
    'Missing YOLOX ball model.',
    'Set TENNIS_COACH_BALL_MODEL_PATH to yolox_nano.onnx and restart the app.',
    `Example: TENNIS_COACH_BALL_MODEL_PATH=${suggestedPath}`,
  ].join('\n')
}

function normalizeHand(value: string | undefined): 'left' | 'right' | null {
  if (!value) return null
  const normalized = value.trim().toLowerCase()
  if (normalized === 'left' || normalized === 'right') {
    return normalized
  }
  return null
}

export function resolvePlayerHandedness(): { player_1: 'left' | 'right'; player_2: 'left' | 'right' } {
  const p1 = normalizeHand(process.env.TENNIS_COACH_PLAYER_1_HAND) ?? 'right'
  const p2 = normalizeHand(process.env.TENNIS_COACH_PLAYER_2_HAND) ?? 'right'
  return { player_1: p1, player_2: p2 }
}

export function resolvePlayerHandednessArgs(): string[] {
  const handedness = resolvePlayerHandedness()
  return [
    '--player-handedness', `player_1=${handedness.player_1}`,
    '--player-handedness', `player_2=${handedness.player_2}`,
  ]
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
  if (/libGLESv2\.so\.2: cannot open shared object file/i.test(detail)) {
    return [
      'Missing system library required by MediaPipe Pose: libGLESv2.so.2',
      'Install it (Ubuntu/WSL): sudo apt update && sudo apt install -y libgles2',
      '',
      detail,
    ].join('\n')
  }
  if (/mediapipe/i.test(detail) && /OSError:/i.test(detail)) {
    return [
      'MediaPipe runtime dependency is missing on this system.',
      'On Ubuntu/WSL, install graphics runtime libs:',
      'sudo apt update && sudo apt install -y libgles2 libgl1',
      '',
      detail,
    ].join('\n')
  }
  return detail
}
