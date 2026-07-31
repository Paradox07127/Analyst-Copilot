/* Marquee timing is a readability decision, not an animation flourish, so the
 * rate is pinned here rather than left to whoever next touches the constants.
 *
 * jsdom performs no layout, so scrollWidth/clientWidth are both 0 and the
 * component would always take the "does not overflow" path. Each test stubs the
 * pair to describe an overflow of a chosen width. */

import { describe, expect, it, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { Marquee } from "../components/ui";

function withOverflow(scrollWidth: number, clientWidth = 100) {
  const scroll = Object.getOwnPropertyDescriptor(
    HTMLElement.prototype,
    "scrollWidth",
  );
  const client = Object.getOwnPropertyDescriptor(
    HTMLElement.prototype,
    "clientWidth",
  );
  Object.defineProperty(HTMLElement.prototype, "scrollWidth", {
    configurable: true,
    get: () => scrollWidth,
  });
  Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    get: () => clientWidth,
  });
  return () => {
    if (scroll) Object.defineProperty(HTMLElement.prototype, "scrollWidth", scroll);
    if (client) Object.defineProperty(HTMLElement.prototype, "clientWidth", client);
  };
}

/** The inner span carries the transform; the outer one is the clipping box. */
function textLayer(): HTMLElement {
  return screen.getByText("a-very-long-dataset-name.csv");
}

function hover() {
  fireEvent.pointerEnter(textLayer().parentElement!);
}

function renderMarquee() {
  render(<Marquee>a-very-long-dataset-name.csv</Marquee>);
}

afterEach(cleanup);

describe("Marquee scroll timing", () => {
  it("does not scroll when the text fits", () => {
    const restore = withOverflow(100, 100);
    try {
      renderMarquee();
      hover();
      const inner = textLayer();
      expect(inner.style.transform).toBe("");
      expect(inner.className).toContain("truncate");
    } finally {
      restore();
    }
  });

  it("travels the full overflow distance, not a fixed amount", () => {
    const restore = withOverflow(340, 100); // 240px hidden
    try {
      renderMarquee();
      hover();
      expect(textLayer().style.transform).toBe("translateX(-240px)");
    } finally {
      restore();
    }
  });

  it("holds ~33px/s so a long name stays readable", () => {
    const restore = withOverflow(340, 100); // 240px over 240*30ms
    try {
      renderMarquee();
      hover();
      const ms = Number(textLayer().style.transitionDuration.replace("ms", ""));
      expect(ms).toBe(7200);
      expect(240 / (ms / 1000)).toBeCloseTo(33.3, 1);
    } finally {
      restore();
    }
  });

  it("floors a tiny overflow so it reads as motion, not a flicker", () => {
    const restore = withOverflow(110, 100); // 10px would be 300ms
    try {
      renderMarquee();
      hover();
      expect(textLayer().style.transitionDuration).toBe("700ms");
    } finally {
      restore();
    }
  });

  it("caps a pathological string so it still finishes", () => {
    const restore = withOverflow(2100, 100); // 2000px would be 60s
    try {
      renderMarquee();
      hover();
      expect(textLayer().style.transitionDuration).toBe("14000ms");
    } finally {
      restore();
    }
  });

  /* An ease-out curve front-loads the travel, so the text moves fastest at the
   * moment you start reading it. Constant speed is the whole point. */
  it("scrolls linearly and keeps the app easing for the snap back", () => {
    const restore = withOverflow(340, 100);
    try {
      renderMarquee();
      hover();
      expect(textLayer().className).toContain("ease-linear");

      fireEvent.pointerLeave(textLayer().parentElement!);
      const inner = textLayer();
      expect(inner.className).toContain("ease-out-quart");
      expect(inner.style.transform).toBe("");
    } finally {
      restore();
    }
  });
});
