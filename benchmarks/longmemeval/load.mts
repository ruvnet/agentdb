/**
 * Streaming loader for the LongMemEval JSON files (ADR-016). longmemeval_m is 2.7 GB, past V8's maximum string
 * length, so the top-level array is split element by element and each instance parsed on its own.
 */
import { createReadStream } from 'node:fs';

export interface Turn {
  role: 'user' | 'assistant';
  content: string;
  has_answer?: boolean;
}

export interface Instance {
  question_id: string;
  question_type: string;
  question: string;
  question_date: string;
  answer: unknown;
  answer_session_ids: string[];
  haystack_session_ids: string[];
  haystack_dates: string[];
  haystack_sessions: Turn[][];
}

/** Yields each element of a top-level JSON array without holding the whole file in one string. */
export async function* streamArray(path: string): AsyncGenerator<Instance> {
  let depth = 0;
  let inString = false;
  let escaped = false;
  let parts: string[] = [];
  let start = -1;
  for await (const chunk of createReadStream(path, { encoding: 'utf8', highWaterMark: 1 << 22 }) as AsyncIterable<string>) {
    for (let i = 0; i < chunk.length; i++) {
      const c = chunk.charCodeAt(i);
      if (inString) {
        if (escaped) escaped = false;
        else if (c === 92 /* \ */) escaped = true;
        else if (c === 34 /* " */) inString = false;
        continue;
      }
      if (c === 34) inString = true;
      else if (c === 123 /* { */) {
        if (++depth === 1) start = i;
      } else if (c === 125 /* } */) {
        if (--depth === 0) {
          parts.push(chunk.slice(start === -1 ? 0 : start, i + 1));
          yield JSON.parse(parts.join('')) as Instance;
          parts = [];
          start = -1;
        }
      }
    }
    if (depth > 0) {
      parts.push(chunk.slice(start === -1 ? 0 : start));
      start = -1;
    }
  }
}

/** Splits text into windows of about `words` words, so no chunk is silently truncated by the embedder. */
export function windows(text: string, words = 160, overlap = 32): string[] {
  const w = text.split(/\s+/).filter(Boolean);
  if (w.length <= words) return [w.join(' ')];
  const out: string[] = [];
  for (let s = 0; s < w.length; s += words - overlap) {
    out.push(w.slice(s, s + words).join(' '));
    if (s + words >= w.length) break;
  }
  return out;
}
