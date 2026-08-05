/* The generated report, rendered. Two presentation-only transforms are
 * applied and nothing else: the exporter's `[Indicative]`-style strength
 * prefixes become badges, and evidence ids the report itself printed become
 * buttons that open the inspector. Figures are left exactly as generated —
 * formatting them here would be rewriting the author's numbers. */

import { Children, isValidElement, useMemo, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import { Link } from "react-router";
import remarkGfm from "remark-gfm";
import { Badge, type Tone } from "../../components/ui";
import { EvidenceLaneBadge } from "../exploration/ExplorationView";
import { headingId, isArtifactIdShape } from "./report-outline";

/* exporter.py:_narrative_claim_text — the only prefixes it emits. Anything
 * else in brackets is the author's own text and is left alone. */
const QUALIFIERS: Record<string, { tone: Tone; meaning: string }> = {
  Indicative: {
    tone: "info",
    meaning:
      "Supported by evidence, but not at the strongest tier. Check the Claim Ledger before quoting it.",
  },
  "Exploratory — hypothesis-generating": {
    tone: "neutral",
    meaning:
      "A lead, not a conclusion. It suggests where to look next; it does not settle anything.",
  },
  "Confirmatory evidence — not a claim of certainty": {
    tone: "info",
    meaning:
      "Evidence from a designated confirmation lane. It still carries limitations and is not a claim of certainty.",
  },
  "Low relevance": {
    tone: "neutral",
    meaning: "Evidence exists but barely bears on the claim.",
  },
  "Unverified figures": {
    tone: "warn",
    meaning:
      "The figures in this claim could not be re-checked against an evidence artifact.",
  },
};

const QUALIFIER_PREFIX = /^\[([^\]]{1,60})\]\s+/;

/* exporter.py:_QUALITY_AGGREGATE_SUFFIX. A grouped limitation names the
 * condition and a few places, then defers the rest to this phrase — which was
 * plain text, so the reader had to go find the page themselves. */
const QUALITY_PAGE_PHRASE = "the Quality page";

export function splitQualifiers(text: string): {
  labels: string[];
  rest: string;
} {
  const labels: string[] = [];
  let rest = text;
  for (;;) {
    const match = QUALIFIER_PREFIX.exec(rest);
    const label = match?.[1];
    if (!label || !(label in QUALIFIERS)) break;
    labels.push(label);
    rest = rest.slice(match[0].length);
  }
  return { labels, rest };
}

function childText(children: ReactNode): string {
  if (typeof children === "string") return children;
  if (typeof children === "number") return String(children);
  if (Array.isArray(children)) return children.map(childText).join("");
  if (isValidElement(children))
    return childText((children.props as { children?: ReactNode }).children);
  return "";
}

function QualifierBadges({ labels }: { labels: string[] }) {
  return (
    <>
      {labels.map((label) => (
        <span key={label} className="mr-1 inline-flex align-baseline">
          {label === "Exploratory — hypothesis-generating" ? (
            <EvidenceLaneBadge lane="exploratory" label={label} />
          ) : label === "Confirmatory evidence — not a claim of certainty" ? (
            <EvidenceLaneBadge lane="confirmatory" label={label} />
          ) : (
            <Badge
              tone={QUALIFIERS[label]?.tone ?? "neutral"}
              title={QUALIFIERS[label]?.meaning}
            >
              {label}
            </Badge>
          )}
        </span>
      ))}
    </>
  );
}

function EvidenceButton({
  artifactId,
  selected,
  onInspect,
}: {
  artifactId: string;
  selected: boolean;
  onInspect: (artifactId: string) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onInspect(artifactId)}
      aria-pressed={selected}
      title="Open this artifact beside the report"
      className={`rounded-sm border px-1 py-0.5 font-mono text-xs hover:border-primary ${
        selected
          ? "border-primary bg-primary/10 text-primary"
          : "border-border text-primary"
      }`}
    >
      {artifactId}
    </button>
  );
}

export function ReportBody({
  markdown,
  inspectableIds,
  selectedId,
  onInspect,
  qualityHref,
}: {
  markdown: string;
  inspectableIds: Set<string>;
  selectedId: string | null;
  onInspect: (artifactId: string) => void;
  /** Where "the Quality page" should go; omitted outside a session route. */
  qualityHref?: string;
}) {
  const components = useMemo<Components>(() => {
    const inspectable = (token: string) =>
      isArtifactIdShape(token) && inspectableIds.has(token);

    const withQualityLink = (children: ReactNode): ReactNode => {
      if (!qualityHref) return children;
      return Children.map(children, (part) => {
        if (typeof part !== "string" || !part.includes(QUALITY_PAGE_PHRASE))
          return part;
        const [before, ...after] = part.split(QUALITY_PAGE_PHRASE);
        return (
          <>
            {before}
            <Link to={qualityHref} className="text-primary hover:underline">
              {QUALITY_PAGE_PHRASE}
            </Link>
            {after.join(QUALITY_PAGE_PHRASE)}
          </>
        );
      });
    };

    const withQualifiers = (children: ReactNode) => {
      /* Children.toArray keys every element, so slicing the array below does
       * not produce an unkeyed list. */
      const parts = Children.toArray(children);
      const first = parts[0];
      if (typeof first !== "string") return children;
      const { labels, rest } = splitQualifiers(first);
      if (labels.length === 0) return children;
      return (
        <>
          <QualifierBadges labels={labels} />
          {rest}
          {parts.slice(1)}
        </>
      );
    };

    return {
      /* Privacy: reports are LLM-generated content in a local-first app, so an
       * <img> would auto-fire an outbound request on page load. */
      img: ({ src, alt }) => {
        const href = typeof src === "string" ? src : undefined;
        return (
          <a href={href} target="_blank" rel="noopener noreferrer">
            {alt || href || "image"}
          </a>
        );
      },
      h1: ({ node, children, ...props }) => (
        <h1 {...props} id={headingId(childText(children))}>
          {children}
        </h1>
      ),
      h2: ({ node, children, ...props }) => (
        <h2 {...props} id={headingId(childText(children))}>
          {children}
        </h2>
      ),
      h3: ({ node, children, ...props }) => (
        <h3 {...props} id={headingId(childText(children))}>
          {children}
        </h3>
      ),
      li: ({ node, children, ...props }) => (
        <li {...props}>{withQualityLink(withQualifiers(children))}</li>
      ),
      td: ({ node, children, ...props }) => {
        const tokens = childText(children)
          .split(",")
          .map((token) => token.trim())
          .filter(Boolean);
        if (tokens.length === 0 || !tokens.every(inspectable))
          return <td {...props}>{children}</td>;
        return (
          <td {...props}>
            <span className="flex flex-wrap gap-1">
              {tokens.map((token) => (
                <EvidenceButton
                  key={token}
                  artifactId={token}
                  selected={token === selectedId}
                  onInspect={onInspect}
                />
              ))}
            </span>
          </td>
        );
      },
      code: ({ node, children, className, ...props }) => {
        const token = childText(children).trim();
        if (!className && inspectable(token))
          return (
            <EvidenceButton
              artifactId={token}
              selected={token === selectedId}
              onInspect={onInspect}
            />
          );
        return (
          <code className={className} {...props}>
            {children}
          </code>
        );
      },
    };
  }, [inspectableIds, selectedId, onInspect, qualityHref]);

  return (
    <article className="report-markdown min-w-0">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {markdown}
      </ReactMarkdown>
    </article>
  );
}
