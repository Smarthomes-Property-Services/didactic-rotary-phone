// functions/lib/googleAi.ts
// Edge (Cloudflare Pages) helper for calling Google's Generative API with an API key.
// Supports two modes:
//  - non-streaming: returns a plain string extracted from the model response
//  - streaming: proxies the upstream streaming Response back to the caller
//
// Usage:
//   // non-streaming
//   const text = await callGoogleGemini(apiKey, prompt, { model: 'models/text-bison-001' })
//
//   // streaming (returns the upstream Response which you should return from the handler)
//   const upstreamResponse = await callGoogleGemini(apiKey, prompt, { model: 'models/text-bison-001', stream: true })
//   return upstreamResponse

export interface GeminiOptions {
  model?: string // model path/name (default 'models/text-bison-001')
  // If 'url' is provided, it is used verbatim instead of composing the default URL
  url?: string
  stream?: boolean // when true, return the upstream Response to be proxied
  temperature?: number
  maxOutputTokens?: number
}

export async function callGoogleGemini(
  apiKey: string,
  prompt: string,
  opts: GeminiOptions = {}
): Promise<string | Response> {
  if (!apiKey) throw new Error('GEMINI_API_KEY required')
  if (!prompt || typeof prompt !== 'string') throw new Error('prompt required')

  const model = opts.model ?? 'models/text-bison-001'
  const baseUrl = opts.url ?? `https://generativelanguage.googleapis.com/v1/${encodeURIComponent(model)}:generate`

  const query = apiKey ? `?key=${encodeURIComponent(apiKey)}` : ''
  const url = `${baseUrl}${query}`

  const body: Record<string, unknown> = {
    prompt: { text: prompt },
    temperature: opts.temperature ?? 0.2,
    maxOutputTokens: opts.maxOutputTokens ?? 512,
  }

  // Streaming mode: request streaming from the upstream API and return the
  // Response object so the caller (the Pages Function handler) can pipe it
  // directly to the client. This preserves SSE or chunked behaviour.
  if (opts.stream) {
    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream, application/json',
      },
      body: JSON.stringify(body),
      // Note: Cloudflare's fetch supports streaming responses; we rely on the
      // caller to return the Response directly to the browser to keep the
      // stream open.
    })

    if (!resp.ok) {
      const text = await resp.text()
      throw new Error(`Google Generative API error: ${resp.status} ${resp.statusText} — ${text}`)
    }

    // Return the upstream Response for direct proxying
    return resp
  }

  // Non-streaming: fetch and extract a single string payload
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify(body),
  })

  const text = await resp.text()
  let data: any
  try { data = JSON.parse(text) } catch { data = text }

  if (!resp.ok) {
    const detail = typeof data === 'object' ? JSON.stringify(data) : String(data)
    throw new Error(`Google Generative API error: ${resp.status} ${resp.statusText} — ${detail}`)
  }

  // Heuristics to extract a user-facing string from common GAi response shapes.
  try {
    // 1) { candidates: [{ content: '...' }...] }
    if (data?.candidates && Array.isArray(data.candidates) && data.candidates[0]) {
      const c = data.candidates[0].content
      if (typeof c === 'string') return c
      if (c?.text) return c.text
      return JSON.stringify(c)
    }

    // 2) { output: [{ content: { text: '...' } }] }
    if (data?.output && Array.isArray(data.output) && data.output[0]) {
      const out = data.output[0]
      if (typeof out === 'string') return out
      if (out?.content && typeof out.content === 'string') return out.content
      if (out?.content?.text) return out.content.text
      return JSON.stringify(out)
    }

    // 3) { predictions: [{ text: '...' }] }
    if (data?.predictions && Array.isArray(data.predictions) && data.predictions[0]?.text) {
      return data.predictions[0].text
    }

    // 4) top-level fields
    if (typeof data === 'object' && (data.text || data.completion || data.outputText)) {
      return data.text ?? data.completion ?? data.outputText
    }

    // 5) fallback to full JSON string
    return typeof data === 'string' ? data : JSON.stringify(data, null, 2)
  } catch (err) {
    return typeof data === 'string' ? data : JSON.stringify(data, null, 2)
  }
}
