type LogoSize = 'large' | 'small'

type LogoProps = {
  size?: LogoSize
  onClick?: () => void
  className?: string
}

/**
 * Brand mark: three connected nodes rising into an arrow —
 * research agents converging on one direction.
 */
function Mark({ className = '' }: { className?: string }) {
  return (
    <div
      className={`flex shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-[#0d9488] to-[#14b8a6] shadow-sm shadow-teal-700/20 ${className}`}
      aria-hidden="true"
    >
      <svg
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="h-[62%] w-[62%]"
      >
        <path
          d="M7.5 24.5 L13.5 17.5 L19.5 11.5"
          stroke="white"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="7.5" cy="24.5" r="2.5" fill="white" />
        <circle cx="13.5" cy="17.5" r="2.5" fill="white" />
        <circle cx="19.5" cy="11.5" r="2.5" fill="white" />
        <path
          d="M19.5 11.5 L19.5 5.2"
          stroke="white"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <path
          d="M15.6 8.4 L19.5 4.2 L23.4 8.4"
          stroke="white"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  )
}

export default function Logo({
  size = 'large',
  onClick,
  className = '',
}: LogoProps) {
  const isLarge = size === 'large'
  const interactive = typeof onClick === 'function'

  const content = isLarge ? (
    <div className={`flex flex-col items-center text-center ${className}`}>
      <Mark className="h-16 w-16 sm:h-[4.5rem] sm:w-[4.5rem]" />
      <p className="mt-4 text-sm font-bold tracking-[0.22em] text-teal-800 uppercase sm:text-base">
        Open Venture Agent
      </p>
      <p className="mt-2 text-[11px] font-medium tracking-[0.18em] text-slate-400 uppercase">
        Research. Validate. Build.
      </p>
    </div>
  ) : (
    <div className={`inline-flex items-center gap-2.5 ${className}`}>
      <Mark className="h-9 w-9" />
      <span className="text-sm font-bold tracking-[0.16em] text-teal-800 uppercase">
        OVA
      </span>
    </div>
  )

  if (!interactive) {
    return content
  }

  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-xl text-left outline-none transition hover:opacity-90 focus-visible:ring-2 focus-visible:ring-teal-500/40"
      aria-label="Open Venture Agent — back to home"
    >
      {content}
    </button>
  )
}
