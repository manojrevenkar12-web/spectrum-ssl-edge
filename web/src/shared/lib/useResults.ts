import { useCallback, useEffect, useState } from 'react';
import { isResults, type Results } from './results';

export type Load =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'empty' }
  | { status: 'ready'; data: Results };

/** Fetch the generated results with loading, error (retry) and empty states. */
export function useResults(url = './results.json'): Load & { retry: () => void } {
  const [state, setState] = useState<Load>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `attempt` is the retry trigger
  useEffect(() => {
    const ctl = new AbortController();
    setState({ status: 'loading' });
    fetch(url, { signal: ctl.signal })
      .then(async (r) => {
        if (r.status === 404) return setState({ status: 'empty' });
        if (!r.ok) return setState({ status: 'error', message: `The results file could not be read (HTTP ${r.status}).` });
        const body: unknown = await r.json().catch(() => null);
        setState(
          isResults(body)
            ? { status: 'ready', data: body }
            : { status: 'error', message: 'The results file is not in the expected format.' },
        );
      })
      .catch((e: Error) => {
        if (e.name !== 'AbortError')
          setState({ status: 'error', message: 'The results file could not be loaded. Check the connection and retry.' });
      });
    return () => ctl.abort();
  }, [url, attempt]);
  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  return { ...state, retry };
}
