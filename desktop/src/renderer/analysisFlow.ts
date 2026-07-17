import { validateTennisAnalysis, type TennisAnalysis } from '../shared/analysis'

export function hasReusableAnalysisReport(value: unknown): value is TennisAnalysis {
  try {
    return validateTennisAnalysis(value).segments.length > 0
  } catch {
    return false
  }
}
