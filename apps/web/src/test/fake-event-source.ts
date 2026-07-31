/* Injectable EventSource double: jsdom has no EventSource, so setup.ts
 * installs this class globally. Tests grab the latest instance and emit
 * typed frames to drive useJobEvents. */

type Listener = (event: MessageEvent) => void;

export class FakeEventSource {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;

  static instances: FakeEventSource[] = [];

  static latest(): FakeEventSource {
    const instance = FakeEventSource.instances.at(-1);
    if (!instance) throw new Error("No FakeEventSource has been constructed");
    return instance;
  }

  static reset(): void {
    FakeEventSource.instances = [];
  }

  readonly url: string;
  readyState: number = FakeEventSource.OPEN;
  onerror: ((event: Event) => void) | null = null;
  private listeners = new Map<string, Set<Listener>>();

  constructor(url: string | URL) {
    this.url = String(url);
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: Listener): void {
    const set = this.listeners.get(type) ?? new Set();
    set.add(listener);
    this.listeners.set(type, set);
  }

  removeEventListener(type: string, listener: Listener): void {
    this.listeners.get(type)?.delete(listener);
  }

  close(): void {
    this.readyState = FakeEventSource.CLOSED;
  }

  emit(type: string, data: unknown): void {
    const message = new MessageEvent(type, { data: JSON.stringify(data) });
    for (const listener of this.listeners.get(type) ?? []) {
      listener(message);
    }
  }

  failFatally(): void {
    this.readyState = FakeEventSource.CLOSED;
    this.onerror?.(new Event("error"));
  }
}

export function installFakeEventSource(): void {
  globalThis.EventSource = FakeEventSource as unknown as typeof EventSource;
}
