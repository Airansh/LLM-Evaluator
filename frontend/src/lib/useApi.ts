"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface ApiState<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  reload: () => void;
}

/**
 * Fetch on mount, and optionally re-fetch every `pollMs` while `shouldPoll(data)` is true
 * (used to follow a run while it executes). A failed poll keeps the last good data.
 */
export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[],
  options: { pollMs?: number; shouldPoll?: (data: T) => boolean } = {},
): ApiState<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const fetcherRef = useRef(fetcher);
  const optionsRef = useRef(options);

  useEffect(() => {
    fetcherRef.current = fetcher;
    optionsRef.current = options;
  });

  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const load = async () => {
      try {
        const result = await fetcherRef.current();
        if (cancelled) return;
        setData(result);
        setError(undefined);
        const { pollMs, shouldPoll } = optionsRef.current;
        if (pollMs && shouldPoll?.(result)) timer = setTimeout(load, pollMs);
      } catch (err) {
        if (!cancelled) setError(err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  return { data, error, loading, reload };
}
