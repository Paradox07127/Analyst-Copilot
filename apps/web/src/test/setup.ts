import "@testing-library/jest-dom/vitest";
import { afterAll, afterEach, beforeAll } from "vitest";
import { cleanup } from "@testing-library/react";
import {
  resetDataOperations,
  resetSettingsState,
  resetSupportDocs,
  resetVerifiedRelations,
} from "./msw/handlers";
import { server } from "./msw/server";
import { FakeEventSource, installFakeEventSource } from "./fake-event-source";

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterAll(() => server.close());

/* jsdom has no EventSource; job SSE tests drive this double directly. */
installFakeEventSource();

/* jsdom's FormData/File live in a different realm than Node's undici fetch,
 * which then fails to serialize multipart bodies ("[object FormData]").
 * Response IS undici's here, so parsing a probe body yields undici's FormData
 * constructor; File comes from node:buffer. */
const probe = new Response(
  '--b\r\nContent-Disposition: form-data; name="x"\r\n\r\n1\r\n--b--\r\n',
  { headers: { "Content-Type": "multipart/form-data; boundary=b" } },
);
const NodeFormData = (await probe.formData()).constructor as typeof FormData;
globalThis.FormData = NodeFormData;
const { File: NodeFile } = await import("node:buffer");
globalThis.File = NodeFile as unknown as typeof File;

/* Node >= 22 defines an experimental globalThis.localStorage (undefined
 * without --localstorage-file) that shadows jsdom's; install a memory-backed
 * Storage so app code can use window.localStorage normally. */
class MemoryStorage {
  private store = new Map<string, string>();
  get length() {
    return this.store.size;
  }
  key(index: number) {
    return [...this.store.keys()][index] ?? null;
  }
  getItem(key: string) {
    return this.store.get(key) ?? null;
  }
  setItem(key: string, value: string) {
    this.store.set(key, String(value));
  }
  removeItem(key: string) {
    this.store.delete(key);
  }
  clear() {
    this.store.clear();
  }
}
Object.defineProperty(window, "localStorage", {
  value: new MemoryStorage(),
  configurable: true,
});

afterEach(() => {
  cleanup();
  server.resetHandlers();
  resetSettingsState();
  resetDataOperations();
  resetSupportDocs();
  resetVerifiedRelations();
  objectUrls.created.length = 0;
  objectUrls.revoked.length = 0;
  window.localStorage.clear();
  delete document.documentElement.dataset["theme"];
  FakeEventSource.reset();
});

/* jsdom lacks both APIs; react-resizable-panels needs ResizeObserver and
 * theme.ts needs matchMedia. */
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver =
  globalThis.ResizeObserver ?? (ResizeObserverStub as typeof ResizeObserver);

/* jsdom lacks IntersectionObserver; chart cards lazy-load on visibility, so
 * the stub reports every observed element as immediately visible to keep the
 * existing test semantics (charts render without scrolling). */
class ImmediatelyVisibleIntersectionObserver {
  constructor(private readonly callback: IntersectionObserverCallback) {}
  observe(target: Element) {
    this.callback(
      [{ isIntersecting: true, target } as IntersectionObserverEntry],
      this as unknown as IntersectionObserver,
    );
  }
  unobserve() {}
  disconnect() {}
  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }
}
globalThis.IntersectionObserver =
  ImmediatelyVisibleIntersectionObserver as unknown as typeof IntersectionObserver;

/* jsdom reports zero offsetWidth/offsetHeight, which makes TanStack Virtual
 * see a 0-height scroll element and render no rows. A fixed viewport-ish size
 * lets virtualized tables render in tests. */
Object.defineProperties(HTMLElement.prototype, {
  offsetWidth: { get: () => 800, configurable: true },
  offsetHeight: { get: () => 600, configurable: true },
});

/* Keep a click's focus from being swallowed by the panel library.
 *
 * react-resizable-panels listens for pointerdown on the *document* and, when
 * it believes the pointer is over a resize handle, calls preventDefault() —
 * which cancels the focus a click would otherwise move. Its hit test is
 * `x >= left - margin && x <= right + margin && ...` (fine margin defaults to
 * 5px), and jsdom reports {0,0,0,0} for every getBoundingClientRect while
 * user-event's synthetic pointer sits at (0,0). So every click "hit" a handle
 * and no input ever received focus: `user.type()` then typed into nothing and
 * silently passed, which is how several tests came to call `user.clear()`
 * first as a workaround.
 *
 * Only the handles are moved off the origin — leaving every other element's
 * rect alone so virtualization measurements are unchanged. */
const RECT_ORIGIN_OFFSET = 100; // > the 5px fine hit-area margin
const originalGetBoundingClientRect =
  Element.prototype.getBoundingClientRect;
Element.prototype.getBoundingClientRect = function getBoundingClientRect(
  this: Element,
): DOMRect {
  if (this.hasAttribute("data-panel-resize-handle-id")) {
    return {
      x: RECT_ORIGIN_OFFSET,
      y: RECT_ORIGIN_OFFSET,
      left: RECT_ORIGIN_OFFSET,
      top: RECT_ORIGIN_OFFSET,
      right: RECT_ORIGIN_OFFSET + 4,
      bottom: RECT_ORIGIN_OFFSET + 600,
      width: 4,
      height: 600,
      toJSON: () => ({}),
    } as DOMRect;
  }
  return originalGetBoundingClientRect.call(this);
};

/* jsdom's AbortSignal is a different realm than Node/undici's Request, which
 * rejects it (react-router passes one on every navigation). Tests never abort
 * navigations, so dropping the signal is safe. */
const OriginalRequest = globalThis.Request;
globalThis.Request = class extends OriginalRequest {
  constructor(input: RequestInfo | URL, init?: RequestInit) {
    if (init?.signal) {
      const { signal: _signal, ...rest } = init;
      super(input, rest);
    } else {
      super(input, init);
    }
  }
};

/* jsdom implements neither object URLs nor a real download, so saveBlob() would
 * throw. Record the calls instead; download tests assert on them. */
export const objectUrls: { created: Blob[]; revoked: string[] } = {
  created: [],
  revoked: [],
};
URL.createObjectURL = (blob: Blob) => {
  objectUrls.created.push(blob);
  return `blob:mock/${objectUrls.created.length}`;
};
URL.revokeObjectURL = (url: string) => {
  objectUrls.revoked.push(url);
};

if (typeof window.matchMedia !== "function") {
  window.matchMedia = (query: string): MediaQueryList =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList;
}
