import axios from 'axios'
import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import Logo from './Logo'

const API_BASE = (import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '')
const API_URL = `${API_BASE}/venture`
const HISTORY_KEY = 'ova.history'
const SESSION_KEY = 'ova.session'
const MAX_HISTORY = 40

const LOADING_MESSAGES = [
  'Researching market...',
  'Analyzing competitors...',
  'Studying customer feedback...',
  'Scoring location...',
  'Building strategy...',
  'Almost done...',
] as const

type OpportunityTier =
  | 'strong_opportunity'
  | 'viable_but_contested'
  | 'insufficient_evidence'
  | 'not_recommended_as_is'
  | string

type NextStep = {
  task_id?: string | null
  phase?: string | null
  title?: string | null
  priority?: string | null
  depends_on?: string[]
  source?: string | null
  status?: string | null
}

type RecommendedStrategy = {
  positioning?: string
  target_customers?: string
  pricing_strategy?: string
  estimated_budget?: string
  key_differentiators?: string[]
}

type KeyFindings = {
  market_research?: string[]
  competitors?: string[]
  customer_pain_points?: string[]
  operational_pain_points?: string[]
  customer_segments?: string[]
  hiring_signal?: string[]
}

type FinalReport = {
  headline: string
  opportunity_tier?: string | null
  confidence?: number | null
  idea_score?: number | null
  idea_score_label?: string | null
  recommended_strategy?: RecommendedStrategy | null
  key_findings?: KeyFindings | string[]
  top_risks?: string[]
  data_limitations?: string[]
  immediate_next_steps?: NextStep[]
  full_analysis?: string | null
}

type ProsConsItem = {
  point?: string
  reasoning?: string
  source?: string
}

type ProsAndCons = {
  pros?: ProsConsItem[]
  cons?: ProsConsItem[]
}

type Strategy = {
  positioning?: string
  target_customers?: string
  pricing_strategy?: string
  estimated_budget?: string | null
  key_differentiators?: string[]
  confidence?: number | null
  idea_score?: number | null
  idea_score_label?: string | null
  assumptions?: string[]
  verification_needed?: string[]
  opportunity_tier?: string | null
  first_call_question?: string | null
  pros_and_cons?: ProsAndCons | null
}

type ResearchFinding = {
  id?: number
  finding: string
  source: string
  source_agent?: string
}

type CompetitorFinding = {
  id?: number
  competitor_name?: string | null
  finding: string
  source: string
  category?: string | null
  source_agent?: string
}

type CustomerSegment = {
  segment_name?: string
  evidence?: string
  needs?: string[]
  source?: string
}

type CustomerPainPoint = {
  pain_point?: string
  source?: string
}

type CustomerResearch = {
  customer_segments?: CustomerSegment[]
  customer_pain_points?: CustomerPainPoint[]
  operational_pain_points?: CustomerPainPoint[]
  confidence?: string
  no_local_data_reason?: string | null
}

type HiringPosting = {
  role?: string
  detail?: string
}

type HiringSignal = {
  postings_found?: Array<HiringPosting | string>
  employer_count?: number
  confidence?: string
  query_used?: string | null
}

type FindingItem = {
  finding: string
  source: string
  source_agent?: string
  competitor_name?: string | null
  category?: string | null
}

type VentureResponse = {
  id?: number
  status?: string
  delivery_mode?: string | null
  business_idea?: string
  findings?: FindingItem[]
  research_findings?: ResearchFinding[]
  competitor_findings?: CompetitorFinding[]
  hiring_signal?: HiringSignal | null
  customer_research?: CustomerResearch | null
  strategy?: Strategy | null
  task_plan?: NextStep[]
  final_report?: FinalReport | null
}

type HistoryItem = {
  id: number
  businessIdea: string
  createdAt: string
}

type PersistedSession = {
  screen: Screen
  businessIdea: string
  venture: VentureResponse | null
}

type Screen = 'input' | 'results' | 'research' | 'strategy' | 'pros_cons' | 'tasks'

const VENTURE_NAV_TABS: { screen: Exclude<Screen, 'input'>; label: string }[] = [
  { screen: 'results', label: 'Report' },
  { screen: 'research', label: 'Research' },
  { screen: 'strategy', label: 'Strategy' },
  { screen: 'pros_cons', label: 'Pros & Cons' },
  { screen: 'tasks', label: 'Tasks' },
]

function isVentureScreen(screen: Screen): screen is Exclude<Screen, 'input'> {
  return screen !== 'input'
}

const TIER_STYLES: Record<string, string> = {
  strong_opportunity: 'bg-emerald-100 text-emerald-800 ring-emerald-200',
  viable_but_contested: 'bg-amber-100 text-amber-900 ring-amber-200',
  insufficient_evidence: 'bg-slate-200 text-slate-700 ring-slate-300',
  not_recommended_as_is: 'bg-rose-100 text-rose-800 ring-rose-200',
}

const PRIORITY_STYLES: Record<string, string> = {
  P0: 'bg-rose-600 text-white',
  P1: 'bg-sky-600 text-white',
  P2: 'bg-slate-500 text-white',
}

const TASK_PHASES = [
  {
    key: 'validate',
    label: 'Validate',
    description: 'Confirm assumptions before committing',
  },
  {
    key: 'decide',
    label: 'Decide',
    description: 'Lock in your final direction',
  },
  {
    key: 'build',
    label: 'Build',
    description: 'Execute the plan',
  },
  {
    key: 'launch',
    label: 'Launch',
    description: 'Go to market',
  },
] as const

function formatTier(tier: string | null | undefined): string {
  if (!tier) return 'Unknown'
  return tier.replace(/_/g, ' ')
}

/** Soft client hint — comma + place, or in/at/near + place-like token. */
function looksLikeHasLocation(idea: string): boolean {
  const text = idea.trim()
  if (!text) return false
  return /,\s*\S+|\b(?:in|at|near)\s+[^\s,]{2,}/i.test(text)
}

const LOCATION_SOFT_WARNING =
  "Your idea might need a location — e.g. 'gifting and event management in Lahore'. You can still submit if this is meant to be a remote/online business."

function formatHistoryDate(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

/** Fix mangled accents (e.g. "Caf" / "CafÃ©" → "Cafe") for already-saved text. */
function sanitizeDisplayText(text: string | null | undefined): string {
  if (!text) return ''
  let cleaned = text
  cleaned = cleaned
    .replace(/CafÃ©/g, 'Cafe')
    .replace(/cafÃ©/g, 'cafe')
    .replace(/CAFÃ‰/g, 'CAFE')
  cleaned = cleaned.replace(/([Cc][Aa][Ff])[\u0080-\u009f\u00e8\u00e9\u00ea\u00eb]/g, '$1e')
  cleaned = cleaned.replace(/[\u0080-\u009f]/g, '')
  cleaned = cleaned.replace(/Café/g, 'Cafe').replace(/café/g, 'cafe').replace(/CAFÉ/g, 'CAFE')
  return cleaned
}

const KEY_FINDING_SECTIONS: {
  key: keyof KeyFindings
  label: string
}[] = [
  { key: 'market_research', label: "What's happening in this industry" },
  { key: 'competitors', label: 'Competitors' },
  { key: 'customer_pain_points', label: 'Customer Pain Points' },
  { key: 'operational_pain_points', label: 'Operational Pain Points' },
  { key: 'customer_segments', label: 'Customer Segments' },
]

function normalizeKeyFindings(
  raw: FinalReport['key_findings'],
): KeyFindings {
  if (!raw) return {}
  if (Array.isArray(raw)) {
    return raw.length > 0 ? { market_research: raw } : {}
  }
  return raw
}

function keyFindingsHaveContent(findings: KeyFindings): boolean {
  return KEY_FINDING_SECTIONS.some(
    ({ key }) => (findings[key]?.length ?? 0) > 0,
  )
}

function loadHistory(): HistoryItem[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as HistoryItem[]
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      (item) =>
        typeof item?.id === 'number' &&
        typeof item?.businessIdea === 'string' &&
        typeof item?.createdAt === 'string',
    )
  } catch {
    return []
  }
}

function saveHistory(items: HistoryItem[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, MAX_HISTORY)))
}

function upsertHistory(items: HistoryItem[], next: HistoryItem): HistoryItem[] {
  const filtered = items.filter((item) => item.id !== next.id)
  return [next, ...filtered].slice(0, MAX_HISTORY)
}

function loadSession(): PersistedSession | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as PersistedSession
    if (
      !parsed ||
      !['input', 'results', 'research', 'strategy', 'tasks'].includes(parsed.screen)
    ) {
      return null
    }
    return parsed
  } catch {
    return null
  }
}

function isHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value.trim())
}

function formatCompetitorCategory(category: string | null | undefined): string | null {
  if (!category) return null
  const normalized = category.trim().toLowerCase()
  if (!normalized || normalized === 'competitor') return null
  if (normalized === 'pricing_intelligence') return 'pricing info'
  if (normalized === 'vendor_technology' || normalized === 'vendor') return 'vendor'
  if (normalized === 'directory_source' || normalized === 'directory') return 'directory'
  return normalized.replace(/_/g, ' ')
}

type CompetitorGroup = 'direct' | 'vendors' | 'other'

function getCompetitorGroup(category: string | null | undefined): CompetitorGroup {
  const normalized = (category || '').trim().toLowerCase()
  if (!normalized || normalized === 'competitor') return 'direct'
  if (normalized === 'vendor' || normalized === 'vendor_technology') return 'vendors'
  if (
    normalized === 'directory' ||
    normalized === 'directory_source' ||
    normalized === 'pricing_intelligence'
  ) {
    return 'other'
  }
  return 'other'
}

function getResearchFindings(venture: VentureResponse | null): ResearchFinding[] {
  if (!venture) return []
  if (venture.research_findings?.length) {
    return venture.research_findings.filter((item) => item?.finding)
  }
  return (venture.findings || [])
    .filter((item) => item.source_agent === 'research' && item.finding)
    .map((item) => ({
      finding: item.finding,
      source: item.source,
      source_agent: item.source_agent,
    }))
}

function getCompetitorFindings(venture: VentureResponse | null): CompetitorFinding[] {
  if (!venture) return []
  if (venture.competitor_findings?.length) {
    return venture.competitor_findings.filter((item) => item?.finding)
  }
  return (venture.findings || [])
    .filter((item) => item.source_agent === 'competitor' && item.finding)
    .map((item) => ({
      finding: item.finding,
      source: item.source,
      competitor_name: item.competitor_name,
      category: item.category,
      source_agent: item.source_agent,
    }))
}

function SourceLink({ source }: { source?: string | null }) {
  const text = (source || '').trim()
  if (!text) {
    return <span className="text-xs text-slate-500">No source</span>
  }
  if (isHttpUrl(text)) {
    return (
      <a
        href={text}
        target="_blank"
        rel="noopener noreferrer"
        className="break-all text-sm font-medium text-teal-700 underline-offset-2 hover:underline"
      >
        View source
      </a>
    )
  }
  return <span className="break-all text-xs text-slate-500">{text}</span>
}

function saveSession(session: PersistedSession) {
  localStorage.setItem(SESSION_KEY, JSON.stringify(session))
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY)
}

function detailToVenture(detail: {
  venture: {
    id: number
    business_idea: string
    status: string
    delivery_mode?: string | null
  }
  research_findings?: ResearchFinding[]
  competitor_findings?: CompetitorFinding[]
  hiring_signal?: HiringSignal | null
  customer_research?: CustomerResearch | null
  strategy?: Strategy | null
  task_plan?: NextStep[]
  final_report?: FinalReport | null
}): VentureResponse {
  return {
    id: detail.venture.id,
    status: detail.venture.status,
    delivery_mode: detail.venture.delivery_mode,
    business_idea: detail.venture.business_idea,
    research_findings: detail.research_findings || [],
    competitor_findings: detail.competitor_findings || [],
    hiring_signal: detail.hiring_signal ?? null,
    customer_research: detail.customer_research ?? null,
    strategy: detail.strategy,
    task_plan: detail.task_plan || [],
    final_report: detail.final_report,
  }
}

function WarningIcon() {
  return (
    <svg
      className="mt-0.5 h-5 w-5 shrink-0 text-amber-600"
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.168 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 6a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 6zm0 8a1 1 0 100-2 1 1 0 000 2z"
        clipRule="evenodd"
      />
    </svg>
  )
}

function AssumptionIcon() {
  return (
    <svg
      className="mt-0.5 h-5 w-5 shrink-0 text-indigo-500"
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
        clipRule="evenodd"
      />
    </svg>
  )
}

function TrashIcon({ className = 'h-4 w-4' }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        d="M8.75 1A2.75 2.75 0 006 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 10.23 1.482l.149-.022.341 10.227A2.75 2.75 0 007.1 18.5h5.8a2.75 2.75 0 002.745-2.322l.34-10.227.15.022a.75.75 0 00.23-1.482A41.03 41.03 0 0014 4.193V3.75A2.75 2.75 0 0011.25 1h-2.5zM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4zM8.58 7.72a.75.75 0 00-1.5.06l.3 7.5a.75.75 0 101.5-.06l-.3-7.5zm4.34.06a.75.75 0 10-1.5-.06l-.3 7.5a.75.75 0 101.5.06l.3-7.5z"
        clipRule="evenodd"
      />
    </svg>
  )
}

function App() {
  const savedSession = loadSession()
  const [screen, setScreen] = useState<Screen>(savedSession?.screen || 'input')
  const [businessIdea, setBusinessIdea] = useState(savedSession?.businessIdea || '')
  const [loading, setLoading] = useState(false)
  const [historyLoadingId, setHistoryLoadingId] = useState<number | null>(null)
  const [historyDeletingId, setHistoryDeletingId] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [messageIndex, setMessageIndex] = useState(0)
  const [venture, setVenture] = useState<VentureResponse | null>(
    savedSession?.venture || null,
  )
  const [history, setHistory] = useState<HistoryItem[]>(() => loadHistory())
  const [historyQuery, setHistoryQuery] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [ideaHasBlurred, setIdeaHasBlurred] = useState(false)
  const [ideaFocused, setIdeaFocused] = useState(false)
  const [debouncedIdea, setDebouncedIdea] = useState(businessIdea)
  const mainScrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    mainScrollRef.current?.scrollTo({ top: 0, left: 0, behavior: 'auto' })
  }, [screen])

  useEffect(() => {
    const id = window.setTimeout(() => setDebouncedIdea(businessIdea), 800)
    return () => window.clearTimeout(id)
  }, [businessIdea])

  useEffect(() => {
    if (!loading) return
    setMessageIndex(0)
    const id = window.setInterval(() => {
      setMessageIndex((i) => (i + 1) % LOADING_MESSAGES.length)
    }, 3500)
    return () => window.clearInterval(id)
  }, [loading])

  useEffect(() => {
    if (screen === 'input' && !venture) {
      clearSession()
      return
    }
    saveSession({
      screen,
      businessIdea,
      venture,
    })
  }, [screen, businessIdea, venture])

  function jumpToScreen(next: Screen) {
    if (next === 'input') return
    setScreen(next)
  }

  function applyVenture(
    next: VentureResponse,
    idea: string,
    nextScreen: Screen = 'results',
  ) {
    const withIdea = { ...next, business_idea: idea }
    setVenture(withIdea)
    setBusinessIdea(idea)
    setScreen(nextScreen)
    setError(null)
    setSidebarOpen(false)

    if (typeof withIdea.id === 'number') {
      setHistory((prev) => {
        const updated = upsertHistory(prev, {
          id: withIdea.id!,
          businessIdea: idea,
          createdAt: new Date().toISOString(),
        })
        saveHistory(updated)
        return updated
      })
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const idea = businessIdea.trim()
    if (!idea || loading) return

    setError(null)
    setLoading(true)
    setHistoryQuery('')

    try {
      const { data } = await axios.post(
        API_URL,
        { business_idea: idea },
        { timeout: 300_000 },
      )
      const response = data as VentureResponse & {
        status?: string
        message?: string
      }
      if (
        response?.status === 'invalid_input' ||
        response?.status === 'missing_location' ||
        response?.status === 'existing_business_detected'
      ) {
        const message =
          response.message?.trim() ||
          (response.status === 'missing_location'
            ? 'This looks like a physical/onsite business — please include a city or area so we can research the local market.'
            : response.status === 'existing_business_detected'
              ? "This looks like it might be naming an existing company rather than describing a new business idea. Try describing your own concept instead — e.g. 'a mid-range clothing brand in Lahore.'"
              : "This doesn't look like a business idea — try describing what you'd like to sell or offer, and ideally where.")
        setVenture(null)
        setScreen('input')
        setError(message)
        return
      }
      if (!response?.final_report?.headline) {
        setScreen('input')
        setError('The API returned no final report. Try again.')
        return
      }
      applyVenture(response, idea, 'results')
    } catch (err) {
      setVenture(null)
      setScreen('input')
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail
        const body = err.response?.data as
          | { status?: string; message?: string }
          | undefined
        if (
          body?.status === 'invalid_input' ||
          body?.status === 'missing_location' ||
          body?.status === 'existing_business_detected'
        ) {
          setError(
            body.message?.trim() ||
              (body.status === 'missing_location'
                ? 'This looks like a physical/onsite business — please include a city or area so we can research the local market.'
                : body.status === 'existing_business_detected'
                  ? "This looks like it might be naming an existing company rather than describing a new business idea. Try describing your own concept instead — e.g. 'a mid-range clothing brand in Lahore.'"
                  : "This doesn't look like a business idea — try describing what you'd like to sell or offer, and ideally where."),
          )
        } else if (typeof detail === 'string') {
          setError(detail)
        } else if (err.code === 'ECONNABORTED') {
          setError('The request timed out. Research can take a few minutes — try again.')
        } else if (!err.response) {
          setError(
            'Could not reach the API at http://127.0.0.1:8000. Is the backend running?',
          )
        } else {
          setError(err.message || 'Something went wrong while starting research.')
        }
      } else if (err instanceof Error) {
        setError(err.message)
      } else {
        setError('Something went wrong while starting research.')
      }
    } finally {
      setLoading(false)
    }
  }

  async function openHistoryItem(item: HistoryItem) {
    if (historyLoadingId !== null || loading) return
    if (venture?.id === item.id && venture.final_report) {
      setBusinessIdea(item.businessIdea)
      setScreen('results')
      setSidebarOpen(false)
      return
    }

    setHistoryLoadingId(item.id)
    setError(null)
    try {
      const { data } = await axios.get(`${API_URL}/${item.id}`, { timeout: 60_000 })
      const mapped = detailToVenture(data)
      if (!mapped.final_report?.headline) {
        throw new Error('This saved venture has no final report yet.')
      }
      applyVenture(mapped, item.businessIdea || mapped.business_idea || '', 'results')
    } catch (err) {
      if (axios.isAxiosError(err)) {
        if (err.response?.status === 404) {
          setError('That saved search no longer exists on the server.')
          setHistory((prev) => {
            const updated = prev.filter((entry) => entry.id !== item.id)
            saveHistory(updated)
            return updated
          })
        } else if (!err.response) {
          setError(
            'Could not reach the API at http://127.0.0.1:8000. Is the backend running?',
          )
        } else {
          setError(err.message || 'Could not load that search.')
        }
      } else if (err instanceof Error) {
        setError(err.message)
      } else {
        setError('Could not load that search.')
      }
    } finally {
      setHistoryLoadingId(null)
    }
  }

  async function deleteHistoryItem(item: HistoryItem) {
    if (historyDeletingId !== null || historyLoadingId !== null || loading) return
    const confirmed = window.confirm(
      `Delete this search?\n\n"${item.businessIdea}"`,
    )
    if (!confirmed) return

    setHistoryDeletingId(item.id)
    try {
      await axios.delete(`${API_URL}/${item.id}`, { timeout: 30_000 })
      setHistory((prev) => {
        const updated = prev.filter((entry) => entry.id !== item.id)
        saveHistory(updated)
        return updated
      })
      if (venture?.id === item.id) {
        startNewResearch()
      }
    } catch (err) {
      if (axios.isAxiosError(err) && err.response?.status === 404) {
        // Already gone on the server — still drop from local history.
        setHistory((prev) => {
          const updated = prev.filter((entry) => entry.id !== item.id)
          saveHistory(updated)
          return updated
        })
        if (venture?.id === item.id) {
          startNewResearch()
        }
        return
      }
      if (axios.isAxiosError(err) && !err.response) {
        setError(
          `Could not reach the API at ${API_BASE}. Is the backend running?`,
        )
      } else if (err instanceof Error) {
        setError(err.message || 'Could not delete that search.')
      } else {
        setError('Could not delete that search.')
      }
    } finally {
      setHistoryDeletingId(null)
    }
  }

  function startNewResearch() {
    setScreen('input')
    setVenture(null)
    setError(null)
    setBusinessIdea('')
    setDebouncedIdea('')
    setIdeaHasBlurred(false)
    setIdeaFocused(false)
    setLoading(false)
    setHistoryQuery('')
    clearSession()
    setSidebarOpen(false)
  }

  function clearHistory() {
    setHistory([])
    saveHistory([])
  }

  const report = venture?.final_report ?? null
  const strategy = venture?.strategy ?? null
  const recommended = report?.recommended_strategy
  const showLocationWarning =
    screen === 'input' &&
    !loading &&
    ideaHasBlurred &&
    !ideaFocused &&
    debouncedIdea.trim().length > 0 &&
    debouncedIdea === businessIdea &&
    !looksLikeHasLocation(debouncedIdea)
  const ideaDescribedBy = [
    error && !loading ? 'idea-error' : null,
    showLocationWarning ? 'idea-location-warning' : null,
  ]
    .filter(Boolean)
    .join(' ')
  const activeId = venture?.id ?? null
  const historyFilter = historyQuery.trim().toLowerCase()
  const filteredHistory =
    historyFilter.length === 0
      ? history
      : history.filter((item) =>
          item.businessIdea.toLowerCase().includes(historyFilter),
        )
  const researchFindings = getResearchFindings(venture)
  const competitorFindings = getCompetitorFindings(venture)
  const directCompetitors = competitorFindings.filter(
    (item) => getCompetitorGroup(item.category) === 'direct',
  )
  const vendorFindings = competitorFindings.filter(
    (item) => getCompetitorGroup(item.category) === 'vendors',
  )
  const otherCompetitorSources = competitorFindings.filter(
    (item) => getCompetitorGroup(item.category) === 'other',
  )
  const customerResearch = venture?.customer_research ?? null
  const customerConfidence = (customerResearch?.confidence || 'none').toLowerCase()
  const customerSegments = customerResearch?.customer_segments || []
  const customerPainPoints = customerResearch?.customer_pain_points || []
  const operationalPainPoints = customerResearch?.operational_pain_points || []
  const taskPlan = venture?.task_plan || []
  const tasksByPhase = TASK_PHASES.map((phase) => ({
    ...phase,
    tasks: taskPlan.filter(
      (task) => (task.phase || '').trim().toLowerCase() === phase.key,
    ),
  }))
  const otherTasks = taskPlan.filter((task) => {
    const phase = (task.phase || '').trim().toLowerCase()
    return !TASK_PHASES.some((entry) => entry.key === phase)
  })

  const resultsTier = (report?.opportunity_tier || strategy?.opportunity_tier || '') as OpportunityTier
  const resultsTierClass =
    TIER_STYLES[resultsTier] || 'bg-slate-200 text-slate-700 ring-slate-300'

  const strategyTier = (strategy?.opportunity_tier || report?.opportunity_tier || '') as OpportunityTier
  const strategyTierClass =
    TIER_STYLES[strategyTier] || 'bg-slate-200 text-slate-700 ring-slate-300'

  const positioning = strategy?.positioning || recommended?.positioning || ''
  const targetCustomers =
    strategy?.target_customers || recommended?.target_customers || ''
  const pricingStrategy =
    strategy?.pricing_strategy || recommended?.pricing_strategy || ''
  const estimatedBudget =
    strategy?.estimated_budget || recommended?.estimated_budget || ''
  const differentiators =
    strategy?.key_differentiators || recommended?.key_differentiators || []
  const strategyConfidence =
    typeof strategy?.confidence === 'number'
      ? strategy.confidence
      : typeof report?.confidence === 'number'
        ? report.confidence
        : null

  return (
    <div className="h-svh overflow-hidden bg-gradient-to-br from-slate-50 via-teal-50/40 to-slate-100 text-slate-800">
      <div className="flex h-full">
        {sidebarOpen && (
          <button
            type="button"
            className="fixed inset-0 z-30 bg-slate-900/30 lg:hidden"
            aria-label="Close history sidebar"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        <aside
          className={`fixed inset-y-0 left-0 z-40 flex h-svh w-72 shrink-0 flex-col border-r border-slate-200 bg-white/95 shadow-xl backdrop-blur transition-transform lg:sticky lg:top-0 lg:z-0 lg:translate-x-0 lg:shadow-none ${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          }`}
        >
          <div className="flex shrink-0 items-center justify-between gap-2 border-b border-slate-100 px-4 py-4">
            <div className="min-w-0">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">
                History
              </p>
              <p className="mt-1 text-sm text-slate-500">Previous searches</p>
            </div>
            <button
              type="button"
              onClick={startNewResearch}
              className="shrink-0 rounded-lg bg-teal-700 px-3 py-1.5 text-xs font-semibold text-white hover:bg-teal-800"
            >
              New
            </button>
          </div>

          <div className="shrink-0 border-b border-slate-100 px-3 py-3">
            <label className="block">
              <span className="sr-only">Search history</span>
              <input
                type="search"
                value={historyQuery}
                onChange={(e) => setHistoryQuery(e.target.value)}
                placeholder="Search history…"
                className="w-full rounded-lg border border-slate-100 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition placeholder:text-slate-500 focus:border-teal-500 focus:bg-white focus:ring-2 focus:ring-teal-500/20"
              />
            </label>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
            {history.length === 0 ? (
              <p className="px-3 py-6 text-sm text-slate-500">
                Your past business ideas will show up here after you run research.
              </p>
            ) : filteredHistory.length === 0 ? (
              <p className="px-3 py-6 text-sm text-slate-500">No matches</p>
            ) : (
              <ul className="space-y-1">
                {filteredHistory.map((item) => {
                  const active = activeId === item.id
                  const loadingItem = historyLoadingId === item.id
                  const deletingItem = historyDeletingId === item.id
                  const busy =
                    historyLoadingId !== null ||
                    historyDeletingId !== null ||
                    loading
                  return (
                    <li key={item.id} className="group relative">
                      <button
                        type="button"
                        onClick={() => openHistoryItem(item)}
                        disabled={busy}
                        className={`w-full rounded-xl py-3 pr-10 pl-3 text-left transition ${
                          active
                            ? 'bg-teal-50 text-teal-950 ring-1 ring-teal-200'
                            : 'text-slate-800 hover:bg-slate-50'
                        } disabled:cursor-wait disabled:opacity-60`}
                      >
                        <span className="line-clamp-2 text-sm font-medium leading-snug">
                          {item.businessIdea}
                        </span>
                        <span className="mt-1 block text-xs text-slate-500">
                          {deletingItem
                            ? 'Deleting…'
                            : loadingItem
                              ? 'Loading…'
                              : formatHistoryDate(item.createdAt)}
                        </span>
                      </button>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation()
                          void deleteHistoryItem(item)
                        }}
                        disabled={busy}
                        aria-label={`Delete search: ${item.businessIdea}`}
                        title="Delete this search"
                        className={`absolute top-2 right-1.5 inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 transition hover:bg-rose-50 hover:text-rose-700 disabled:cursor-wait disabled:opacity-40 max-lg:opacity-100 lg:opacity-0 lg:group-hover:opacity-100 lg:group-focus-within:opacity-100 ${
                          deletingItem ? 'opacity-100' : ''
                        }`}
                      >
                        <TrashIcon />
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
          </div>

          {history.length > 0 && (
            <div className="shrink-0 border-t border-slate-100 p-3">
              <button
                type="button"
                onClick={clearHistory}
                className="w-full rounded-lg px-3 py-2 text-xs font-medium text-slate-500 hover:bg-slate-50 hover:text-slate-800"
              >
                Clear history
              </button>
            </div>
          )}
        </aside>

        <div
          ref={mainScrollRef}
          className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto"
        >
          <div className="sticky top-0 z-20 flex items-center gap-2 border-b border-slate-200/70 bg-white/80 px-3 py-3 backdrop-blur lg:hidden">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700"
            >
              <span aria-hidden="true">☰</span>
              History
            </button>
          </div>

          <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4 py-6 sm:px-6 lg:px-8 lg:pt-2">
            <header className="mb-6 flex justify-center sm:mb-8">
              <Logo
                size="large"
                onClick={screen === 'input' ? undefined : startNewResearch}
              />
            </header>

            {isVentureScreen(screen) && venture && (
              <nav
                className="sticky top-[3.25rem] z-10 -mx-4 mb-8 border-b border-slate-200/80 bg-slate-50/90 px-4 backdrop-blur sm:-mx-6 sm:px-6 lg:top-0 lg:-mx-8 lg:px-8"
                aria-label="Venture sections"
              >
                <div className="flex items-end justify-between gap-3">
                  <div
                    className="flex min-w-0 flex-1 gap-1 overflow-x-auto"
                    role="tablist"
                  >
                    {VENTURE_NAV_TABS.map((tab) => {
                      const active = screen === tab.screen
                      return (
                        <button
                          key={tab.screen}
                          type="button"
                          role="tab"
                          aria-selected={active}
                          onClick={() => jumpToScreen(tab.screen)}
                          className={`shrink-0 border-b-2 px-3 py-3 text-sm transition sm:px-4 sm:text-base ${
                            active
                              ? 'border-teal-200 font-bold text-teal-900'
                              : 'border-transparent font-medium text-slate-500 hover:border-slate-300 hover:text-slate-800'
                          }`}
                        >
                          {tab.label}
                        </button>
                      )
                    })}
                  </div>
                  <button
                    type="button"
                    onClick={startNewResearch}
                    className="mb-2 shrink-0 rounded-lg px-2.5 py-1.5 text-xs font-semibold text-slate-500 transition hover:bg-white/80 hover:text-teal-800 sm:text-sm"
                  >
                    New Research
                  </button>
                </div>
              </nav>
            )}

            {error && screen !== 'input' && (
              <div
                className="mb-6 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800"
                role="alert"
              >
                {error}
              </div>
            )}

            {screen === 'input' && (
              <main className="flex flex-1 items-center justify-center">
                <div className="w-full max-w-xl rounded-2xl border border-slate-200/80 bg-white/90 p-8 shadow-lg shadow-slate-200/60 backdrop-blur">
                  <h1 className="text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
                    What venture are you exploring?
                  </h1>
                  <p className="mt-3 text-base text-slate-500">
                    Describe your business idea and location. We&apos;ll research the
                    market, competitors, customers, and strategy.
                  </p>

                  <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
                    <label className="block">
                      <span className="mb-2 block text-sm font-medium text-slate-700">
                        Business idea
                      </span>
                      <input
                        type="text"
                        value={businessIdea}
                        onChange={(e) => {
                          setBusinessIdea(e.target.value)
                          if (error) setError(null)
                        }}
                        onFocus={() => setIdeaFocused(true)}
                        onBlur={() => {
                          setIdeaFocused(false)
                          setIdeaHasBlurred(true)
                        }}
                        disabled={loading}
                        placeholder="e.g. bakery in Bahria Town, Islamabad"
                        aria-invalid={Boolean(error && !loading)}
                        aria-describedby={ideaDescribedBy || undefined}
                        className={`w-full rounded-xl border bg-white px-4 py-3 text-base text-slate-900 outline-none transition placeholder:text-slate-500 focus:ring-2 disabled:cursor-not-allowed disabled:bg-slate-50 ${
                          error && !loading
                            ? 'border-rose-400 focus:border-rose-500 focus:ring-rose-500/20'
                            : showLocationWarning
                              ? 'border-amber-400 focus:border-amber-500 focus:ring-amber-500/20'
                              : 'border-slate-300 focus:border-teal-500 focus:ring-teal-500/20'
                        }`}
                        autoFocus
                      />
                      {error && !loading && (
                        <p
                          id="idea-error"
                          className="mt-2 text-sm leading-relaxed text-rose-700"
                          role="alert"
                        >
                          {error}
                        </p>
                      )}
                      {showLocationWarning && (
                        <p
                          id="idea-location-warning"
                          className="mt-2 text-sm leading-relaxed text-amber-800"
                          role="status"
                        >
                          {LOCATION_SOFT_WARNING}
                        </p>
                      )}
                    </label>

                    <button
                      type="submit"
                      disabled={loading || !businessIdea.trim()}
                      className="inline-flex w-full items-center justify-center rounded-xl bg-teal-700 px-5 py-3 text-base font-semibold text-white transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                    >
                      {loading ? 'Researching…' : 'Start Research'}
                    </button>
                  </form>

                  {loading && (
                    <div
                      className="mt-6 rounded-xl border border-teal-100 bg-teal-50/80 px-4 py-4 text-center"
                      role="status"
                      aria-live="polite"
                    >
                      <div className="mx-auto mb-3 h-8 w-8 animate-spin rounded-full border-2 border-teal-200 border-t-teal-700" />
                      <p className="text-sm font-medium text-teal-900">
                        {LOADING_MESSAGES[messageIndex]}
                      </p>
                      <p className="mt-1 text-xs text-teal-700/80">
                        This usually takes 1–2 minutes. Keep this tab open.
                      </p>
                    </div>
                  )}
                </div>
              </main>
            )}

            {screen === 'results' && report && (
              <main className="flex-1 space-y-6 pb-8">
                <div className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-lg shadow-slate-200/50 sm:p-8">
                  {typeof (report.idea_score ?? strategy?.idea_score) === 'number' && (
                    <div className="mb-6">
                      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-teal-700">
                        Idea score
                      </p>
                      <p className="mt-2 text-5xl font-bold tracking-tight text-slate-900 sm:text-6xl">
                        {report.idea_score ?? strategy?.idea_score}
                        <span className="text-2xl font-semibold text-slate-500 sm:text-3xl">
                          /100
                        </span>
                      </p>
                      <p className="mt-2 text-base font-medium text-slate-500 sm:text-lg">
                        {report.idea_score_label ||
                          strategy?.idea_score_label ||
                          'Score based on research signals'}
                      </p>
                    </div>
                  )}

                  <div className="flex flex-wrap items-center gap-3">
                    <span
                      className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold capitalize ring-1 ring-inset ${resultsTierClass}`}
                    >
                      {formatTier(report.opportunity_tier)}
                    </span>
                    {typeof report.confidence === 'number' && (
                      <span className="text-sm font-medium text-slate-500">
                        {report.confidence}% confidence
                      </span>
                    )}
                  </div>

                  <h1 className="mt-5 text-2xl font-semibold leading-snug tracking-tight text-slate-900 sm:text-3xl">
                    {report.headline}
                  </h1>
                  {(venture?.business_idea || businessIdea) && (
                    <p className="mt-3 text-sm text-slate-500">
                      {venture?.business_idea || businessIdea}
                    </p>
                  )}
                </div>

                <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">Key findings</h2>
                  {(() => {
                    const findings = normalizeKeyFindings(report.key_findings)
                    if (!keyFindingsHaveContent(findings)) {
                      return (
                        <p className="mt-3 text-sm text-slate-500">
                          No key findings returned.
                        </p>
                      )
                    }
                    return (
                      <div className="mt-5 space-y-6">
                        {KEY_FINDING_SECTIONS.map(({ key, label }) => {
                          const items = findings[key] ?? []
                          if (items.length === 0) return null
                          return (
                            <div key={key}>
                              <h3 className="font-bold text-slate-900">
                                {label}
                              </h3>
                              <ul className="mt-2 list-disc space-y-2 pl-5 text-slate-700">
                                {items.map((finding) => (
                                  <li key={finding} className="leading-relaxed">
                                    {sanitizeDisplayText(finding)}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )
                        })}
                      </div>
                    )
                  })()}
                </section>

                <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">Top risks</h2>
                  {(report.top_risks?.length ?? 0) > 0 ? (
                    <ul className="mt-4 space-y-3">
                      {report.top_risks!.map((risk) => (
                        <li
                          key={risk}
                          className="flex gap-3 rounded-xl border border-amber-100 bg-amber-50/60 px-3 py-3 text-slate-800"
                        >
                          <WarningIcon />
                          <span className="leading-relaxed">{risk}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-3 text-sm text-slate-500">No top risks returned.</p>
                  )}
                </section>

                {(report.data_limitations?.length ?? 0) > 0 && (
                  <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                    <h2 className="text-lg font-semibold text-slate-900">
                      Data limitations
                    </h2>
                    <ul className="mt-4 space-y-3">
                      {report.data_limitations!.map((item) => (
                        <li
                          key={item}
                          className="rounded-xl border border-slate-100 bg-slate-50/60 px-4 py-3 text-sm leading-relaxed text-slate-700"
                        >
                          {sanitizeDisplayText(item)}
                        </li>
                      ))}
                    </ul>
                  </section>
                )}

                <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">
                    Immediate next steps
                  </h2>
                  {(report.immediate_next_steps?.length ?? 0) > 0 ? (
                    <ul className="mt-4 space-y-3">
                      {report.immediate_next_steps!.map((step, index) => {
                        const priority = (step.priority || '').toUpperCase()
                        const priorityClass =
                          PRIORITY_STYLES[priority] || 'bg-slate-400 text-white'
                        return (
                          <li
                            key={step.task_id || `${step.title}-${index}`}
                            className="flex items-start gap-3 rounded-xl border border-slate-100 bg-slate-50/80 px-3 py-3"
                          >
                            <input
                              type="checkbox"
                              className="mt-1 h-4 w-4 rounded border-slate-300 text-teal-600 focus:ring-teal-600"
                              aria-label={step.title || 'Next step'}
                            />
                            <div className="min-w-0 flex-1">
                              <div className="flex flex-wrap items-center gap-2">
                                {priority && (
                                  <span
                                    className={`inline-flex rounded-md px-2 py-0.5 text-[11px] font-bold tracking-wide ${priorityClass}`}
                                  >
                                    {priority}
                                  </span>
                                )}
                                {step.phase && (
                                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
                                    {step.phase}
                                  </span>
                                )}
                              </div>
                              <p className="mt-1 text-slate-800 leading-relaxed">
                                {step.title}
                              </p>
                            </div>
                          </li>
                        )
                      })}
                    </ul>
                  ) : (
                    <p className="mt-3 text-sm text-slate-500">
                      No immediate next steps returned.
                    </p>
                  )}
                </section>
              </main>
            )}

            {screen === 'research' && venture && (
              <main className="flex-1 space-y-6 pb-8">
                <div className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-lg shadow-slate-200/50 sm:p-8">
                  <p className="text-sm font-medium uppercase tracking-wide text-slate-500">
                    Research
                  </p>
                  <h1 className="mt-3 text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
                    What we found
                  </h1>
                  {(venture.business_idea || businessIdea) && (
                    <p className="mt-3 text-sm text-slate-500">
                      {venture.business_idea || businessIdea}
                    </p>
                  )}
                </div>

                <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">
                    What&apos;s happening in this industry
                  </h2>
                  {researchFindings.length > 0 ? (
                    <ul className="mt-4 space-y-3">
                      {researchFindings.map((item, index) => (
                        <li
                          key={item.id ?? `${item.finding}-${index}`}
                          className="rounded-xl border border-slate-100 bg-slate-50/80 px-4 py-4"
                        >
                          <p className="leading-relaxed text-slate-800">
                            {sanitizeDisplayText(item.finding)}
                          </p>
                          <div className="mt-3">
                            <SourceLink source={item.source} />
                          </div>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-3 text-sm text-slate-500">
                      No market research findings returned.
                    </p>
                  )}
                </section>

                <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">Competitors</h2>
                  {competitorFindings.length === 0 ? (
                    <p className="mt-3 text-sm text-slate-500">
                      No competitor findings returned.
                    </p>
                  ) : (
                    <div className="mt-4 space-y-6">
                      {directCompetitors.length > 0 && (
                        <div>
                          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                            Direct Competitors
                          </h3>
                          <ul className="mt-3 space-y-3">
                            {directCompetitors.map((item, index) => (
                              <li
                                key={
                                  item.id ??
                                  `direct-${item.competitor_name}-${item.finding}-${index}`
                                }
                                className="rounded-xl border border-slate-100 bg-slate-50/80 px-4 py-4"
                              >
                                {item.competitor_name && (
                                  <h4 className="text-base font-semibold text-slate-900">
                                    {sanitizeDisplayText(item.competitor_name)}
                                  </h4>
                                )}
                                <p
                                  className={`${item.competitor_name ? 'mt-2' : ''} leading-relaxed text-slate-800`}
                                >
                                  {sanitizeDisplayText(item.finding)}
                                </p>
                                <div className="mt-3">
                                  <SourceLink source={item.source} />
                                </div>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {vendorFindings.length > 0 && (
                        <div>
                          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                            Tools &amp; Vendors
                          </h3>
                          <ul className="mt-3 space-y-3">
                            {vendorFindings.map((item, index) => {
                              const categoryLabel = formatCompetitorCategory(item.category)
                              return (
                                <li
                                  key={
                                    item.id ??
                                    `vendor-${item.competitor_name}-${item.finding}-${index}`
                                  }
                                  className="rounded-xl border border-slate-100/80 bg-slate-50/50 px-4 py-3"
                                >
                                  <div className="flex flex-wrap items-center gap-2">
                                    {item.competitor_name && (
                                      <h4 className="text-sm font-medium text-slate-900">
                                        {sanitizeDisplayText(item.competitor_name)}
                                      </h4>
                                    )}
                                    {categoryLabel && (
                                      <span className="inline-flex rounded-md bg-slate-200/60 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                                        {categoryLabel}
                                      </span>
                                    )}
                                  </div>
                                  <p className="mt-1.5 text-sm leading-relaxed text-slate-800">
                                    {sanitizeDisplayText(item.finding)}
                                  </p>
                                  <div className="mt-2">
                                    <SourceLink source={item.source} />
                                  </div>
                                </li>
                              )
                            })}
                          </ul>
                        </div>
                      )}

                      {otherCompetitorSources.length > 0 && (
                        <div>
                          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                            Other Sources
                          </h3>
                          <ul className="mt-3 space-y-3">
                            {otherCompetitorSources.map((item, index) => {
                              const categoryLabel = formatCompetitorCategory(item.category)
                              return (
                                <li
                                  key={
                                    item.id ??
                                    `other-${item.competitor_name}-${item.finding}-${index}`
                                  }
                                  className="rounded-xl border border-slate-100 bg-slate-50/80 px-4 py-4"
                                >
                                  <div className="flex flex-wrap items-center gap-2">
                                    {item.competitor_name && (
                                      <h4 className="text-base font-semibold text-slate-900">
                                        {sanitizeDisplayText(item.competitor_name)}
                                      </h4>
                                    )}
                                    {categoryLabel && (
                                      <span className="inline-flex rounded-md bg-slate-200/80 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                                        {categoryLabel}
                                      </span>
                                    )}
                                  </div>
                                  <p
                                    className={`${item.competitor_name || categoryLabel ? 'mt-2' : ''} leading-relaxed text-slate-800`}
                                  >
                                    {sanitizeDisplayText(item.finding)}
                                  </p>
                                  <div className="mt-3">
                                    <SourceLink source={item.source} />
                                  </div>
                                </li>
                              )
                            })}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </section>

                <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">
                    Customer Insights
                  </h2>
                  {customerConfidence === 'none' ? (
                    <p className="mt-3 leading-relaxed text-slate-700">
                      {customerResearch?.no_local_data_reason ||
                        'No local customer voice was available for this idea.'}
                    </p>
                  ) : (
                    <div className="mt-4 space-y-6">
                      <div>
                        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                          Customer segments
                        </h3>
                        {customerSegments.length > 0 ? (
                          <ul className="mt-3 space-y-3">
                            {customerSegments.map((segment, index) => (
                              <li
                                key={`${segment.segment_name}-${index}`}
                                className="rounded-xl border border-slate-100 bg-slate-50/80 px-4 py-4"
                              >
                                <h4 className="text-base font-semibold text-slate-900">
                                  {segment.segment_name || 'Unnamed segment'}
                                </h4>
                                {(segment.needs?.length ?? 0) > 0 && (
                                  <ul className="mt-3 list-disc space-y-1 pl-5 text-slate-700">
                                    {segment.needs!.map((need) => (
                                      <li key={need} className="leading-relaxed">
                                        {need}
                                      </li>
                                    ))}
                                  </ul>
                                )}
                                {segment.source && (
                                  <div className="mt-3">
                                    <SourceLink source={segment.source} />
                                  </div>
                                )}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="mt-2 text-sm text-slate-500">
                            No customer segments returned.
                          </p>
                        )}
                      </div>

                      <div>
                        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                          Customer pain points
                        </h3>
                        {customerPainPoints.length > 0 ? (
                          <ul className="mt-3 space-y-3">
                            {customerPainPoints.map((item, index) => (
                              <li
                                key={`${item.pain_point}-${index}`}
                                className="rounded-xl border border-amber-100 bg-amber-50/60 px-4 py-4"
                              >
                                <p className="leading-relaxed text-slate-800">
                                  {item.pain_point}
                                </p>
                                {item.source && (
                                  <div className="mt-3">
                                    <SourceLink source={item.source} />
                                  </div>
                                )}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="mt-2 text-sm text-slate-500">
                            No customer pain points returned.
                          </p>
                        )}
                      </div>

                      <div>
                        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                          Operational pain points
                        </h3>
                        {operationalPainPoints.length > 0 ? (
                          <ul className="mt-3 space-y-3">
                            {operationalPainPoints.map((item, index) => (
                              <li
                                key={`${item.pain_point}-${index}`}
                                className="rounded-xl border border-slate-100 bg-slate-50/60 px-4 py-4"
                              >
                                <p className="leading-relaxed text-slate-800">
                                  {item.pain_point}
                                </p>
                                {item.source && (
                                  <div className="mt-3">
                                    <SourceLink source={item.source} />
                                  </div>
                                )}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="mt-2 text-sm text-slate-500">
                            No operational pain points returned.
                          </p>
                        )}
                      </div>
                    </div>
                  )}
                </section>
              </main>
            )}

            {screen === 'strategy' && venture && (
              <main className="flex-1 space-y-6 pb-8">
                <div className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-lg shadow-slate-200/50 sm:p-8">
                  <p className="text-sm font-medium uppercase tracking-wide text-slate-500">
                    Full strategy
                  </p>
                  <div className="mt-3 flex flex-wrap items-center gap-3">
                    <span
                      className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold capitalize ring-1 ring-inset ${strategyTierClass}`}
                    >
                      {formatTier(strategyTier)}
                    </span>
                    {strategyConfidence !== null && (
                      <span className="text-sm font-medium text-slate-500">
                        {strategyConfidence}% confidence
                      </span>
                    )}
                  </div>
                </div>

                {strategy?.first_call_question && (
                  <section className="rounded-2xl border border-teal-700 bg-teal-50/90 p-6 shadow-md shadow-teal-100/50 sm:p-8">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-teal-700">
                      First call question
                    </p>
                    <h2 className="mt-3 text-xl font-semibold leading-snug text-teal-950 sm:text-2xl">
                      {strategy.first_call_question}
                    </h2>
                    <p className="mt-2 text-sm text-teal-800/80">
                      The single most important thing to verify in person before you
                      commit.
                    </p>
                  </section>
                )}

                <section className="rounded-2xl border border-teal-100 bg-gradient-to-br from-white to-teal-50/70 p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">Positioning</h2>
                  <p className="mt-3 text-base leading-relaxed text-slate-800">
                    {positioning || 'No positioning returned.'}
                  </p>
                </section>

                <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">Target customers</h2>
                  <p className="mt-3 leading-relaxed text-slate-700">
                    {targetCustomers || 'No target customers returned.'}
                  </p>
                </section>

                <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">Pricing strategy</h2>
                  <p className="mt-3 leading-relaxed text-slate-700">
                    {pricingStrategy || 'No pricing strategy returned.'}
                  </p>
                </section>

                <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">
                    Estimated startup budget
                  </h2>
                  <p className="mt-1 text-xs font-medium uppercase tracking-wide text-amber-700/90">
                    Rough estimate — verify locally
                  </p>
                  <p className="mt-3 leading-relaxed text-slate-700">
                    {estimatedBudget || 'No estimated budget returned.'}
                  </p>
                </section>

                <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">
                    What makes you stand out
                  </h2>
                  {differentiators.length > 0 ? (
                    <ul className="mt-4 list-disc space-y-2 pl-5 text-slate-700">
                      {differentiators.map((item) => (
                        <li key={item} className="leading-relaxed">
                          {item}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-3 text-sm text-slate-500">
                      No differentiators returned.
                    </p>
                  )}
                </section>

                <section className="rounded-2xl border border-indigo-100 bg-indigo-50/70 p-6 shadow-md shadow-indigo-100/40 sm:p-8">
                  <div className="flex items-center gap-2">
                    <AssumptionIcon />
                    <h2 className="text-lg font-semibold text-indigo-950">Assumptions</h2>
                  </div>
                  <p className="mt-2 text-sm text-indigo-800/80">
                    Beliefs to treat carefully — not yet verified facts.
                  </p>
                  {(strategy?.assumptions?.length ?? 0) > 0 ? (
                    <ul className="mt-4 space-y-3">
                      {strategy!.assumptions!.map((item) => (
                        <li
                          key={item}
                          className="flex gap-3 rounded-xl border border-indigo-100 bg-white/70 px-3 py-3 text-slate-800"
                        >
                          <AssumptionIcon />
                          <span className="leading-relaxed">{item}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-3 text-sm text-indigo-700/70">
                      No assumptions returned.
                    </p>
                  )}
                </section>

                <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                  <h2 className="text-lg font-semibold text-slate-900">
                    Verification needed
                  </h2>
                  {(strategy?.verification_needed?.length ?? 0) > 0 ? (
                    <ul className="mt-4 space-y-3">
                      {strategy!.verification_needed!.map((item) => (
                        <li
                          key={item}
                          className="flex gap-3 rounded-xl border border-amber-100 bg-amber-50/60 px-3 py-3 text-slate-800"
                        >
                          <WarningIcon />
                          <span className="leading-relaxed">{item}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-3 text-sm text-slate-500">
                      No verification items returned.
                    </p>
                  )}
                </section>
              </main>
            )}

            {screen === 'pros_cons' && venture && (
              <main className="flex-1 space-y-6 pb-8">
                <div className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-lg shadow-slate-200/50 sm:p-8">
                  <p className="text-sm font-medium uppercase tracking-wide text-slate-500">
                    Pros &amp; Cons
                  </p>
                  <h1 className="mt-3 text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
                    What works in your favor — and what doesn&apos;t
                  </h1>
                  {(venture.business_idea || businessIdea) && (
                    <p className="mt-3 text-sm text-slate-500">
                      {venture.business_idea || businessIdea}
                    </p>
                  )}
                  <p className="mt-3 text-sm text-slate-500">
                    Each point is grounded in the research for this idea, not
                    generic advice.
                  </p>
                </div>

                {(() => {
                  const bundle = strategy?.pros_and_cons
                  const pros = bundle?.pros || []
                  const cons = bundle?.cons || []
                  if (pros.length === 0 && cons.length === 0) {
                    return (
                      <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                        <p className="text-sm text-slate-500">
                          No pros and cons returned yet. Run a new research to
                          generate this section.
                        </p>
                      </section>
                    )
                  }
                  return (
                    <div className="grid gap-6 lg:grid-cols-2">
                      <section className="rounded-2xl border border-emerald-100 bg-emerald-50/40 p-6 shadow-md shadow-slate-200/40 sm:p-8">
                        <h2 className="text-lg font-semibold text-emerald-950">
                          Pros
                        </h2>
                        {pros.length > 0 ? (
                          <ul className="mt-4 space-y-4">
                            {pros.map((item, index) => (
                              <li
                                key={`${item.point}-${index}`}
                                className="rounded-xl border border-emerald-100/80 bg-white/80 px-4 py-3"
                              >
                                <p className="font-semibold text-emerald-950">
                                  {sanitizeDisplayText(item.point) || 'Pro'}
                                </p>
                                {item.reasoning && (
                                  <p className="mt-2 text-sm leading-relaxed text-slate-700">
                                    {sanitizeDisplayText(item.reasoning)}
                                  </p>
                                )}
                                {item.source && (
                                  <p className="mt-2 text-xs leading-relaxed break-all text-slate-500">
                                    Source: {sanitizeDisplayText(item.source)}
                                  </p>
                                )}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="mt-3 text-sm text-slate-500">
                            No pros returned.
                          </p>
                        )}
                      </section>

                      <section className="rounded-2xl border border-rose-100 bg-rose-50/40 p-6 shadow-md shadow-slate-200/40 sm:p-8">
                        <h2 className="text-lg font-semibold text-rose-700">
                          Cons
                        </h2>
                        {cons.length > 0 ? (
                          <ul className="mt-4 space-y-4">
                            {cons.map((item, index) => (
                              <li
                                key={`${item.point}-${index}`}
                                className="rounded-xl border border-rose-100/80 bg-white/80 px-4 py-3"
                              >
                                <p className="font-semibold text-rose-800">
                                  {sanitizeDisplayText(item.point) || 'Con'}
                                </p>
                                {item.reasoning && (
                                  <p className="mt-2 text-sm leading-relaxed text-slate-700">
                                    {sanitizeDisplayText(item.reasoning)}
                                  </p>
                                )}
                                {item.source && (
                                  <p className="mt-2 text-xs leading-relaxed break-all text-slate-500">
                                    Source: {sanitizeDisplayText(item.source)}
                                  </p>
                                )}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="mt-3 text-sm text-slate-500">
                            No cons returned.
                          </p>
                        )}
                      </section>
                    </div>
                  )
                })()}
              </main>
            )}

            {screen === 'tasks' && venture && (
              <main className="flex-1 space-y-6 pb-8">
                <div className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-lg shadow-slate-200/50 sm:p-8">
                  <p className="text-sm font-medium uppercase tracking-wide text-slate-500">
                    Task plan
                  </p>
                  <h1 className="mt-3 text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
                    Your next moves
                  </h1>
                  {(venture.business_idea || businessIdea) && (
                    <p className="mt-3 text-sm text-slate-500">
                      {venture.business_idea || businessIdea}
                    </p>
                  )}
                </div>

                {taskPlan.length === 0 ? (
                  <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                    <p className="text-sm text-slate-500">No task plan returned.</p>
                  </section>
                ) : (
                  <>
                    {tasksByPhase.map((phase) => (
                      <section
                        key={phase.key}
                        className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8"
                      >
                        <h2 className="text-lg font-semibold text-slate-900">
                          {phase.label}
                        </h2>
                        <p className="mt-1 text-sm text-slate-500">{phase.description}</p>
                        {phase.tasks.length > 0 ? (
                          <ul className="mt-4 space-y-3">
                            {phase.tasks.map((task, index) => {
                              const priority = (task.priority || '').toUpperCase()
                              const priorityClass =
                                PRIORITY_STYLES[priority] || 'bg-slate-400 text-white'
                              const dependsOn = (task.depends_on || []).filter(Boolean)
                              return (
                                <li
                                  key={task.task_id || `${phase.key}-${task.title}-${index}`}
                                  className="flex items-start gap-3 rounded-xl border border-slate-100 bg-slate-50/80 px-3 py-3"
                                >
                                  <input
                                    type="checkbox"
                                    className="mt-1 h-4 w-4 rounded border-slate-300 text-teal-600 focus:ring-teal-600"
                                    aria-label={task.title || 'Task'}
                                  />
                                  <div className="min-w-0 flex-1">
                                    <div className="flex flex-wrap items-center gap-2">
                                      {priority && (
                                        <span
                                          className={`inline-flex rounded-md px-2 py-0.5 text-[11px] font-bold tracking-wide ${priorityClass}`}
                                        >
                                          {priority}
                                        </span>
                                      )}
                                      {task.task_id && (
                                        <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
                                          {task.task_id}
                                        </span>
                                      )}
                                    </div>
                                    <p className="mt-1 text-slate-800 leading-relaxed">
                                      {sanitizeDisplayText(task.title || '')}
                                    </p>
                                    {dependsOn.length > 0 && (
                                      <p className="mt-1 text-xs text-slate-500">
                                        Depends on: {dependsOn.join(', ')}
                                      </p>
                                    )}
                                  </div>
                                </li>
                              )
                            })}
                          </ul>
                        ) : (
                          <p className="mt-3 text-sm text-slate-500">
                            No {phase.label.toLowerCase()} tasks in this plan.
                          </p>
                        )}
                      </section>
                    ))}

                    {otherTasks.length > 0 && (
                      <section className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-md shadow-slate-200/40 sm:p-8">
                        <h2 className="text-lg font-semibold text-slate-900">Other</h2>
                        <p className="mt-1 text-sm text-slate-500">
                          Tasks outside the standard phases
                        </p>
                        <ul className="mt-4 space-y-3">
                          {otherTasks.map((task, index) => {
                            const priority = (task.priority || '').toUpperCase()
                            const priorityClass =
                              PRIORITY_STYLES[priority] || 'bg-slate-400 text-white'
                            const dependsOn = (task.depends_on || []).filter(Boolean)
                            return (
                              <li
                                key={task.task_id || `other-${task.title}-${index}`}
                                className="flex items-start gap-3 rounded-xl border border-slate-100 bg-slate-50/80 px-3 py-3"
                              >
                                <input
                                  type="checkbox"
                                  className="mt-1 h-4 w-4 rounded border-slate-300 text-teal-600 focus:ring-teal-600"
                                  aria-label={task.title || 'Task'}
                                />
                                <div className="min-w-0 flex-1">
                                  <div className="flex flex-wrap items-center gap-2">
                                    {priority && (
                                      <span
                                        className={`inline-flex rounded-md px-2 py-0.5 text-[11px] font-bold tracking-wide ${priorityClass}`}
                                      >
                                        {priority}
                                      </span>
                                    )}
                                    {task.phase && (
                                      <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
                                        {task.phase}
                                      </span>
                                    )}
                                  </div>
                                  <p className="mt-1 text-slate-800 leading-relaxed">
                                    {sanitizeDisplayText(task.title || '')}
                                  </p>
                                  {dependsOn.length > 0 && (
                                    <p className="mt-1 text-xs text-slate-500">
                                      Depends on: {dependsOn.join(', ')}
                                    </p>
                                  )}
                                </div>
                              </li>
                            )
                          })}
                        </ul>
                      </section>
                    )}
                  </>
                )}
              </main>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
