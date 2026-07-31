/* Latching viewport visibility: flips to true the first time the element
 * intersects and stays true, so a lazy fetch fires once per card. */

import { useEffect, useRef, useState } from "react";

export function useInView<T extends Element>() {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    if (inView) return;
    const element = ref.current;
    if (!element) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) setInView(true);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [inView]);

  return { ref, inView };
}
