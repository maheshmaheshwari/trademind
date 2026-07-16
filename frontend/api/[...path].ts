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
// Path handling: /api/* passes through as-is; /auth/* is rewritten by
// vercel.json to /api/auth/* so it reaches this function, and is mapped
// back to /auth/* here before forwarding.

const SPACE_URL = (
  process.env.SPACE_URL || 'https://maheshmaheshwari-trademind.hf.space'
).replace(/\/+$/, '');

export const config = { api: { bodyParser: false } };

export default async function handler(req: any, res: any) {
  const hfToken = process.env.HF_SPACE_TOKEN;
  if (!hfToken) {
    res.status(500).json({ error: 'HF_SPACE_TOKEN is not configured' });
    return;
  }

  let path: string = req.url || '';
  if (path.startsWith('/api/auth/') || path === '/api/auth') {
    path = path.slice('/api'.length); // rewritten /auth/* call — restore it
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
