/* Shared surface, status and disclosure primitives.
 *
 * Before this file the app hand-wrote `rounded-base border border-border` at
 * 200 call sites, which meant every container sat at exactly one elevation and
 * nothing could be visually subordinate to anything else. These give the pages
 * a three-step hierarchy (page > section > detail) and one place to fix it. */

import {
  useEffect,
  useId,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type HTMLAttributes,
  type ReactNode,
  type Ref,
} from "react";

/* ---------------------------------------------------------------- surfaces */

type CardTone = "default" | "quiet" | "raised" | "critical" | "warn" | "brand";

const CARD_TONE: Record<CardTone, string> = {
  default: "border-border bg-bg",
  /* Recessed: for detail that sits *inside* another card without adding a
   * second frame around it. Nested borders are the main source of the
   * boxed-in-a-box look on the dense pages. */
  quiet: "border-transparent bg-surface",
  raised: "border-border bg-bg shadow-card",
  critical: "border-status-critical/40 bg-status-critical/5",
  warn: "border-status-warn/45 bg-status-warn/5",
  brand: "border-primary/35 bg-primary/5",
};

/* `as` exists so a card can keep the element semantics its content deserves —
 * an <article> for a dataset, an <li> inside a list — instead of forcing a
 * wrapper div around every row. */
export function Card({
  as: Tag = "div",
  tone = "default",
  className = "",
  ref,
  children,
  ...rest
}: {
  as?: "div" | "article" | "section" | "li";
  tone?: CardTone;
  className?: string;
  /* React 19 passes ref as a plain prop; the dialog focus trap needs one. */
  ref?: Ref<HTMLElement>;
  children: ReactNode;
} & Omit<HTMLAttributes<HTMLElement>, "className" | "children">) {
  /* The `as` union makes the intrinsic ref types intersect to something no
   * single element satisfies; the rendered tag is still Tag at runtime. */
  const Element = Tag as "div";
  return (
    <Element
      ref={ref as Ref<HTMLDivElement>}
      className={`rounded-base border ${CARD_TONE[tone]} ${className}`}
      {...rest}
    >
      {children}
    </Element>
  );
}

/* ----------------------------------------------------------------- buttons */

/* Four weights, in the order a screen may use them: at most one `primary` per
 * view, `secondary` for the alternatives beside it, `ghost` for anything that
 * only reveals or navigates, `danger` for the irreversible confirm. Before this
 * the same action was solid on one page and outlined on the next, so weight
 * carried no information. */
export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

const BUTTON_VARIANT: Record<ButtonVariant, string> = {
  primary: "bg-primary text-bg hover:opacity-90",
  secondary: "border border-border bg-bg hover:bg-surface",
  ghost: "text-status-neutral hover:bg-surface hover:text-text",
  danger:
    "border border-status-critical/50 text-status-critical hover:bg-status-critical/10",
};

const BUTTON_SIZE = {
  sm: "px-2 py-1 text-xs",
  md: "px-3 py-1.5 text-sm",
} as const;

/** The class string, so a `Link`/`NavLink` can wear the same button as a real
 *  `<button>` without a polymorphic wrapper.
 *
 *  `className` ADDS to the variant; it cannot reliably override it. Tailwind
 *  resolves conflicts by generated-CSS order, not by order in the attribute,
 *  and this project has no tailwind-merge — so `ghost` + `hover:text-…` loses
 *  to the variant's own `hover:text-text`, silently. If you need different
 *  hover/display behaviour, pick another variant, wrap the control, or add a
 *  variant here. Do not pass a competing utility and assume it wins. */
export function buttonClass({
  variant = "secondary",
  size = "md",
  className = "",
}: {
  variant?: ButtonVariant;
  size?: keyof typeof BUTTON_SIZE;
  className?: string;
} = {}): string {
  /* `transition` rather than `transition-colors`: the press scale needs
   * transform in the property list. 2% is under the threshold where it reads as
   * a bounce and still lands as acknowledgement on a control whose result can
   * be a minute away. */
  return `inline-flex w-fit shrink-0 items-center justify-center gap-1.5 rounded-base font-medium transition duration-fast ease-out-quart active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50 disabled:active:scale-100 ${BUTTON_VARIANT[variant]} ${BUTTON_SIZE[size]} ${className}`;
}

export function Button({
  variant = "secondary",
  size = "md",
  className = "",
  type = "button",
  children,
  ...rest
}: {
  variant?: ButtonVariant;
  size?: keyof typeof BUTTON_SIZE;
  className?: string;
  children: ReactNode;
} & Omit<ButtonHTMLAttributes<HTMLButtonElement>, "className" | "children">) {
  return (
    <button type={type} className={buttonClass({ variant, size, className })} {...rest}>
      {children}
    </button>
  );
}

/** Square control holding one glyph. `label` is required because an icon-only
 *  control has no accessible name of its own. */
export function IconButton({
  label,
  variant = "ghost",
  className = "",
  children,
  ...rest
}: {
  label: string;
  variant?: ButtonVariant;
  className?: string;
  children: ReactNode;
} & Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  "className" | "children" | "aria-label" | "title"
>) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={`inline-flex size-7 shrink-0 items-center justify-center rounded-base transition duration-fast ease-out-quart active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50 disabled:active:scale-100 ${BUTTON_VARIANT[variant]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

/** Title + optional supporting line + optional right-aligned actions. */
export function SectionHeader({
  title,
  description,
  actions,
  level = 2,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  level?: 1 | 2 | 3;
}) {
  const Heading = `h${level}` as "h1" | "h2" | "h3";
  const size =
    level === 1 ? "text-xl font-semibold" : level === 2 ? "text-base font-semibold" : "text-sm font-semibold";

  return (
    <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-1">
      <div className="flex min-w-0 flex-col gap-1">
        <Heading className={size}>{title}</Heading>
        {description && (
          <p className="max-w-content text-sm text-status-neutral">{description}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ status */

export type Tone = "neutral" | "info" | "ok" | "warn" | "critical" | "llm" | "brand";

const BADGE_SOFT: Record<Tone, string> = {
  neutral: "bg-code-bg text-status-neutral",
  info: "bg-status-info/15 text-status-info",
  ok: "bg-status-ok/15 text-status-ok",
  warn: "bg-status-warn/15 text-status-warn",
  critical: "bg-status-critical/15 text-status-critical",
  llm: "bg-status-llm/15 text-status-llm",
  brand: "bg-primary/15 text-primary",
};

const BADGE_OUTLINE: Record<Tone, string> = {
  neutral: "border border-border text-status-neutral",
  info: "border border-status-info/40 text-status-info",
  ok: "border border-status-ok/40 text-status-ok",
  warn: "border border-status-warn/40 text-status-warn",
  critical: "border border-status-critical/40 text-status-critical",
  llm: "border border-status-llm/40 text-status-llm",
  brand: "border border-primary/40 text-primary",
};

/* `caps` is deliberately opt-in. Shouting every attribute in small caps is what
 * makes the question cards unreadable — five equal-weight chips and no way to
 * tell the run outcome from the question's genre. Reserve caps for state. */
export function Badge({
  tone = "neutral",
  variant = "soft",
  caps = false,
  title,
  children,
}: {
  tone?: Tone;
  variant?: "soft" | "outline";
  caps?: boolean;
  title?: string;
  children: ReactNode;
}) {
  const style = variant === "soft" ? BADGE_SOFT[tone] : BADGE_OUTLINE[tone];
  return (
    <span
      title={title}
      /* w-fit: as a direct child of a flex column the default `stretch`
       * alignment makes an inline-flex badge span the full width. */
      className={`inline-flex w-fit shrink-0 items-center gap-1 rounded-sm px-1.5 py-0.5 text-xs leading-tight font-medium ${
        caps ? "tracking-wide uppercase" : ""
      } ${style}`}
    >
      {children}
    </span>
  );
}

const DOT_TONE: Record<Tone, string> = {
  neutral: "bg-status-neutral",
  info: "bg-status-info",
  ok: "bg-status-ok",
  warn: "bg-status-warn",
  critical: "bg-status-critical",
  llm: "bg-status-llm",
  brand: "bg-primary",
};

const DOT_RING: Record<Tone, string> = {
  neutral: "ring-status-neutral/30",
  info: "ring-status-info/30",
  ok: "ring-status-ok/30",
  warn: "ring-status-warn/30",
  critical: "ring-status-critical/30",
  llm: "ring-status-llm/30",
  brand: "ring-primary/30",
};

/* `motion` answers two different questions that used to share one `pulse` flag,
 * which is why an upload in flight and a preflight row blocked on the user
 * animated identically. A blocked thing must not look busy: `attention` draws a
 * static halo, `working` breathes. */
export function Dot({
  tone,
  motion,
}: {
  tone: Tone;
  motion?: "working" | "attention";
}) {
  return (
    <span
      aria-hidden
      className={`inline-block h-2 w-2 shrink-0 rounded-full ${DOT_TONE[tone]} ${
        motion === "working"
          ? "animate-breathe"
          : motion === "attention"
            ? `ring-2 ring-offset-1 ring-offset-bg ${DOT_RING[tone]}`
            : ""
      }`}
    />
  );
}

/* ------------------------------------------------------------------ text */

/** Ellipsis at rest; on hover or keyboard focus the full string slides past so a
 *  clipped name stays readable without a tooltip.
 *
 *  Two layers because one cannot do both jobs: `text-overflow: ellipsis` needs
 *  the text element to be the thing overflowing, while a transform needs it to
 *  be at its natural width. At rest the inner span is `truncate` (100% wide, so
 *  the ellipsis renders); while scrolling it switches to `inline-block` at
 *  natural width and translates inside the clipping parent.
 *
 *  Measured on pointer-enter, not on render: the shift is only knowable after
 *  layout and most rows are never hovered. `scrollWidth` reports full content
 *  width even while clipped, which is what makes the resting-state measurement
 *  correct. */
/* Reading speed, not animation speed. ~33px/s: slow enough to follow a long
 * dataset name to the end without re-hovering. The outbound leg is deliberately
 * `linear` — an ease-out curve front-loads the travel, so the text was moving
 * fastest at exactly the moment you start reading it, which is what made the
 * first attempt unreadable even before the rate came down. */
const MARQUEE_MS_PER_PX = 30;
/** Floor, so a two-character overflow is a movement rather than a flicker. */
const MARQUEE_MIN_MS = 700;
/** Ceiling, so a pathologically long string still finishes in a sane time. */
const MARQUEE_MAX_MS = 14_000;

export function Marquee({
  children,
  className = "",
  title,
}: {
  children: ReactNode;
  className?: string;
  /** Falls back to a native tooltip; the text itself is always fully in the DOM
   *  so assistive tech never sees the clipped form. */
  title?: string;
}) {
  const innerRef = useRef<HTMLSpanElement>(null);
  const [shift, setShift] = useState(0);

  const start = () => {
    const el = innerRef.current;
    if (!el) return;
    const overflow = el.scrollWidth - el.clientWidth;
    if (overflow > 1) setShift(overflow);
  };
  const stop = () => setShift(0);

  return (
    <span
      onPointerEnter={start}
      onPointerLeave={stop}
      onFocus={start}
      onBlur={stop}
      className={`block min-w-0 overflow-hidden ${className}`}
    >
      <span
        ref={innerRef}
        /* On the inner span, not the clipping parent: callers and tests expect
         * the tooltip to belong to the element that holds the text. */
        title={title}
        style={
          shift
            ? {
                transform: `translateX(-${shift}px)`,
                transitionDuration: `${Math.min(
                  MARQUEE_MAX_MS,
                  Math.max(MARQUEE_MIN_MS, shift * MARQUEE_MS_PER_PX),
                )}ms`,
              }
            : undefined
        }
        /* Linear only on the way out. Snapping back is not reading, so the
         * return leg keeps the app's own ease-out at the default duration. */
        className={`transition-transform ${
          shift
            ? "ease-linear inline-block whitespace-nowrap"
            : "ease-out-quart block truncate"
        }`}
      >
        {children}
      </span>
    </span>
  );
}

/* ----------------------------------------------------------------- metrics */

/** One figure with its label. `hint` explains what the figure actually counts —
 *  the tiles that read "Findings recorded 7 / not the Findings page count" are
 *  the reason this is a first-class slot rather than a caption. */
export function MetricTile({
  label,
  value,
  hint,
  tone = "neutral",
  emphasis = false,
  title,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  tone?: Tone;
  emphasis?: boolean;
  /** Where a figure's basis needs a full sentence the tile has no room for —
   *  what a cost estimate excludes, why two counts of "findings" differ. */
  title?: string;
}) {
  const valueTone =
    tone === "neutral" || !emphasis
      ? ""
      : {
          info: "text-status-info",
          ok: "text-status-ok",
          warn: "text-status-warn",
          critical: "text-status-critical",
          llm: "text-status-llm",
          brand: "text-primary",
          neutral: "",
        }[tone];

  return (
    <div title={title} className="flex min-w-0 flex-col gap-0.5">
      {/* Mono, small-caps, tracked out: the label is the axis name, not prose,
        * and setting it apart from the figure is what lets the strip drop its
        * per-tile borders and still read as discrete tiles. */}
      {/* Plain truncate, not Marquee: these are short fixed axis names that do
        * not overflow in practice, and wrapping them puts an extra element
        * between the label and its value that callers walk with parentElement. */}
      <span className="block truncate font-mono text-[10.5px] tracking-[0.08em] text-status-neutral uppercase">
        {label}
      </span>
      <span className={`tabular text-lg leading-tight font-semibold ${valueTone}`}>
        {value}
      </span>
      {hint && (
        <span className="text-xs leading-snug text-status-neutral">{hint}</span>
      )}
    </div>
  );
}

/** Metric tiles as one banded strip instead of N separate boxes — a row of
 *  identically-sized bordered cards is the flattest possible reading of a
 *  summary, and it competes with the real content below it. */
export function MetricStrip({ children }: { children: ReactNode }) {
  return (
    <Card tone="quiet" className="flex flex-wrap gap-x-8 gap-y-3 px-4 py-3">
      {children}
    </Card>
  );
}

/* ------------------------------------------------------------- disclosure */

/** Collapsible region. Animates on grid-template-rows so the height transition
 *  does not need a measured pixel value and never animates layout width. */
export function Disclosure({
  summary,
  meta,
  defaultOpen = false,
  children,
}: {
  summary: ReactNode;
  meta?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const panelId = useId();

  return (
    <div className="flex flex-col">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        aria-controls={panelId}
        className="flex items-center gap-2 rounded-base py-1 text-left text-sm hover:text-primary"
      >
        <Chevron open={open} />
        <span className="font-medium">{summary}</span>
        {meta && (
          <span className="ml-auto text-xs text-status-neutral">{meta}</span>
        )}
      </button>
      <div
        id={panelId}
        className="grid transition-[grid-template-rows] duration-base ease-out-quart"
        style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          {/* `inert` rather than `hidden`: it takes the collapsed content out
           * of the a11y tree and the tab order without setting display:none,
           * which would cancel the row transition above. */}
          <div inert={!open} aria-hidden={!open} className="pt-2">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}

export function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`shrink-0 transition-transform duration-base ease-out-quart ${
        open ? "rotate-0" : "-rotate-90"
      }`}
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

/* ------------------------------------------------------------------ staged */

/* Staged disclosure, deliberately NOT `Disclosure`: these are stops every user
 * walks in order, not optional depth most users skip (NN/g draws that line and
 * says the two need different components). Drawing the whole chain before stop
 * one is what stops a primary button reading as irreversible — the review stop
 * is visible before it is reached. Each feature owns its own step vocabulary. */
export function StepChain({
  label,
  steps,
  current = 0,
}: {
  label: string;
  steps: readonly string[];
  current?: number;
}) {
  return (
    <ol
      aria-label={label}
      className="flex flex-wrap items-center gap-x-1 gap-y-1 text-xs"
    >
      {steps.map((step, index) => {
        const done = index < current;
        const active = index === current;
        return (
          <li key={step} className="flex items-center gap-1">
            {index > 0 && (
              <span aria-hidden className="text-status-neutral/50">
                →
              </span>
            )}
            <span
              aria-current={active ? "step" : undefined}
              className={`flex items-center gap-1 rounded-sm px-1.5 py-0.5 ${
                active
                  ? "bg-primary/15 font-medium text-primary"
                  : done
                    ? "text-status-ok"
                    : "text-status-neutral"
              }`}
            >
              <span
                className={`tabular flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] ${
                  active
                    ? "bg-primary text-bg"
                    : done
                      ? "bg-status-ok/20"
                      : "bg-track"
                }`}
              >
                {index + 1}
              </span>
              {step}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

/* ----------------------------------------------------------------- jargon */

/** Keep in step with the `w-64` on the Hint panel below. */
const PANEL_WIDTH_PX = 256;

/** Inline definition for a term the product invented. The alternative that was
 *  in place — a paragraph of explanation under every control — is what makes
 *  these pages exhausting to read. Explain on demand, not pre-emptively. */
export function Hint({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  /* Flipped at open time rather than fixed by the caller: these sit in the top
   * bar and in table headers, where a left-anchored 16rem panel runs off the
   * right edge. Measuring beats asking every call site to know where it is. */
  const [alignRight, setAlignRight] = useState(false);
  const panelId = useId();
  const wrapRef = useRef<HTMLSpanElement>(null);

  const toggle = () => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (rect) setAlignRight(rect.left + PANEL_WIDTH_PX > window.innerWidth);
    setOpen((current) => !current);
  };

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    const onPointerDown = (event: PointerEvent) => {
      if (!wrapRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open]);

  return (
    <span ref={wrapRef} className="relative inline-flex">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        aria-controls={panelId}
        aria-label={`What is ${label}?`}
        className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-border text-[10px] leading-none text-status-neutral hover:border-primary hover:text-primary"
      >
        ?
      </button>
      {open && (
        <span
          id={panelId}
          role="note"
          className={`animate-enter absolute top-5 z-50 w-64 rounded-base border border-border bg-bg p-2.5 text-xs leading-relaxed font-normal text-text shadow-overlay ${
            alignRight ? "right-0" : "left-0"
          }`}
        >
          <span className="mb-1 block font-semibold">{label}</span>
          {children}
        </span>
      )}
    </span>
  );
}

/* --------------------------------------------------------------- formatting */

/** 1_000_163 -> "1.0M". Full value belongs in a title attribute, not the cell. */
export function formatCompact(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) < 1000) return String(value);
  if (Math.abs(value) < 1_000_000) return `${(value / 1000).toFixed(1)}k`;
  if (Math.abs(value) < 1_000_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  return `${(value / 1_000_000_000).toFixed(1)}B`;
}

/** Seconds -> "1m 53s" / "204ms". Stage durations span 0.05s to several
 *  minutes, and a bare "113.2" seconds reads as a measurement, not a wait. */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  if (mins < 60) return rest === 0 ? `${mins}m` : `${mins}m ${rest}s`;
  return `${Math.floor(mins / 60)}h ${mins % 60}m`;
}

/** 0.1234 -> "12.3%". */
export function formatPercent(ratio: number, digits = 1): string {
  if (!Number.isFinite(ratio)) return "—";
  return `${(ratio * 100).toFixed(digits)}%`;
}
