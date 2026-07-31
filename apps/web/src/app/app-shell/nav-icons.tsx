import type { ReactNode } from "react";

/* Hand-drawn 24px stroke glyphs standing in for the Material Symbols the
 * session navigation. Inline rather than a
 * package: the app ships no icon dependency and this nav is the only caller. */
const GLYPHS = {
  dashboard: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="3" width="7" height="7" rx="1" />
      <rect x="3" y="14" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
    </>
  ),
  table: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M3 9h18M9 9v11" />
    </>
  ),
  rule: (
    <>
      <path d="M4 6h8M4 11h5" />
      <path d="M12.5 16.5l2.5 2.5 5-5" />
    </>
  ),
  chart: <path d="M5 20V10M12 20V4M19 20v-7" />,
  cleaning: (
    <>
      <path d="M14.5 3.5l6 6" />
      <path d="M13 8l3 3-5.5 5.5a2.5 2.5 0 0 1-3.5 0l0 0a2.5 2.5 0 0 1 0-3.5z" />
    </>
  ),
  hub: (
    <>
      <circle cx="12" cy="12" r="2.5" />
      <circle cx="5" cy="5" r="2" />
      <circle cx="19" cy="5" r="2" />
      <circle cx="12" cy="20" r="2" />
      <path d="M10.3 10.3L6.4 6.4M13.7 10.3l3.9-3.9M12 14.5V18" />
    </>
  ),
  book: (
    <>
      <path d="M3 5.5A2.5 2.5 0 0 1 5.5 3H11v16H5.5A2.5 2.5 0 0 0 3 21.5z" />
      <path d="M21 5.5A2.5 2.5 0 0 0 18.5 3H13v16h5.5a2.5 2.5 0 0 1 2.5 2.5z" />
    </>
  ),
  quiz: (
    <>
      <rect x="3" y="3" width="18" height="18" rx="2.5" />
      <path d="M9.6 9.4a2.5 2.5 0 1 1 3.1 2.4c-.6.2-1 .8-1 1.4v.4" />
      <path d="M11.7 17h.01" />
    </>
  ),
  analytics: (
    <>
      <path d="M4 4v16h16" />
      <path d="M8 16v-4M12.5 16V8M17 16v-6" />
    </>
  ),
  factCheck: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M7 9h6M7 13h3" />
      <path d="M13.8 15.6l2 2 3.4-3.4" />
    </>
  ),
  compare: (
    <>
      <path d="M3 9h13l-3.5-3.5" />
      <path d="M21 15H8l3.5 3.5" />
    </>
  ),
  chat: <path d="M21 12a8 8 0 0 1-8 8H4l2.2-2.7A8 8 0 1 1 21 12z" />,
  sparkle: (
    <>
      <path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z" />
      <path d="M18.4 16.4l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z" />
    </>
  ),
  description: (
    <>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
      <path d="M9 13h6M9 17h4" />
    </>
  ),
  timeline: (
    <>
      <path d="M6 4v16" />
      <circle cx="6" cy="8" r="2" />
      <circle cx="6" cy="16" r="2" />
      <path d="M10 8h9M10 16h6" />
    </>
  ),
  box: (
    <>
      <path d="M3 8l9-5 9 5v8l-9 5-9-5z" />
      <path d="M3 8l9 5 9-5M12 13v8" />
    </>
  ),
  board: (
    <>
      <rect x="3" y="4" width="5" height="16" rx="1" />
      <rect x="9.5" y="4" width="5" height="11" rx="1" />
      <rect x="16" y="4" width="5" height="14" rx="1" />
    </>
  ),
  rocket: (
    <>
      <path d="M12 2s4.5 2.5 4.5 8c0 3-1.5 5-1.5 5h-6s-1.5-2-1.5-5C7.5 4.5 12 2 12 2z" />
      <circle cx="12" cy="9" r="1.5" />
      <path d="M9 16l-2.5 3 3-1M15 16l2.5 3-3-1" />
    </>
  ),
  settings: (
    <path
      fill="currentColor"
      stroke="none"
      fillRule="evenodd"
      d="M19.43 12.98c.04-.32.07-.65.07-.98s-.02-.66-.07-.98l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46a.5.5 0 0 0-.6-.22l-2.49 1a7.42 7.42 0 0 0-1.69-.98L14.5 2.42A.5.5 0 0 0 14 2h-4a.5.5 0 0 0-.5.42l-.38 2.65c-.61.25-1.17.59-1.69.98l-2.49-1a.5.5 0 0 0-.6.22l-2 3.46a.5.5 0 0 0 .12.64l2.11 1.65c-.04.32-.08.65-.08.98s.03.66.08.98l-2.11 1.65a.5.5 0 0 0-.12.64l2 3.46a.5.5 0 0 0 .6.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.04.24.25.42.5.42h4c.25 0 .46-.18.5-.42l.38-2.65c.61-.25 1.17-.58 1.69-.98l2.49 1a.5.5 0 0 0 .6-.22l2-3.46a.5.5 0 0 0-.12-.64l-2.11-1.65ZM12 15.5A3.5 3.5 0 1 1 12 8a3.5 3.5 0 0 1 0 7.5Z"
    />
  ),
} satisfies Record<string, ReactNode>;

export type IconName = keyof typeof GLYPHS;

export function NavIcon({ name }: { name: IconName }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="shrink-0"
    >
      {GLYPHS[name]}
    </svg>
  );
}
