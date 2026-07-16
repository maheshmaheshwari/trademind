// Vercel serverless proxy → private Hugging Face Space backend.
//
// The Space is private, so every request to it must carry an HF token in the
// Authorization header. That token lives only in Vercel env vars
// (HF_SPACE_TOKEN — use a fine-grained READ token scoped to the Space, not
// the write token used for deploys) and is added here, server-side. It is
// never sent to or visible in the browser.
//
// The user's app JWT arrives from the browser in Authorization; since that
// header is taken by the HF token, we forward the JWT as
// X-App-Authorization. The backend (api/server.py promote_proxied_auth
// middleware) moves it back into Authorization before routing.
//
// Routing: Vercel only supports single-segment dynamic API filenames outside
// Next.js (a [...path].ts catch-all does NOT match /api/a/b), so vercel.json
// rewrites every /api/* and /auth/* request to this one function, passing
// the original path in the __proxy_path query param as a fallback in case
// req.url arrives as the rewrite destination rather than the original URL.

const SPACE_URL = (
  process.env.SPACE_URL || 'https://maheshmaheshwari-trademind.hf.space'
).replace(/\/+$/, '');

export const config = { api: { bodyParser: false } };

function upstreamPath(reqUrl: string): string {
  const u = new URL(reqUrl, 'http://internal');
  let path = u.searchParams.get('__proxy_path');
  if (path) {
    u.searchParams.delete('__proxy_path');
  } else {
    path = u.pathname; // req.url kept the original requested path
  }
  const qs = u.searchParams.toString();
  return path + (qs ? `?${qs}` : '');
}

export default async function handler(req: any, res: any) {
  const hfToken = process.env.HF_SPACE_TOKEN;
  if (!hfToken) {
    res.status(500).json({ error: 'HF_SPACE_TOKEN is not configured' });
    return;
  }

  const path = upstreamPath(req.url || '');
  if (!path.startsWith('/api/') && !path.startsWith('/auth/')) {
    res.status(404).json({ error: 'Not found' });
    return;
  }

  const headers: Record<string, string> = {
    authorization: `Bearer ${hfToken}`,
  };
  if (req.headers['content-type']) {
    headers['content-type'] = req.headers['content-type'];
  }
  if (req.headers['authorization']) {
    headers['x-app-authorization'] = req.headers['authorization'];
  }

  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = Buffer.concat(chunks);

  let upstream: Response;
  try {
    upstream = await fetch(SPACE_URL + path, {
      method: req.method,
      headers,
      body: req.method === 'GET' || req.method === 'HEAD' ? undefined : body,
    });
  } catch {
    res.status(502).json({ error: 'Backend unreachable' });
    return;
  }

  res.status(upstream.status);
  const contentType = upstream.headers.get('content-type');
  if (contentType) res.setHeader('content-type', contentType);
  res.send(Buffer.from(await upstream.arrayBuffer()));
}
