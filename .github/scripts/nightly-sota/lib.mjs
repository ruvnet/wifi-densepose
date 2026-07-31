// SPDX-License-Identifier: MIT
import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

export const EVIDENCE_SCHEMA = 'ruview.nightly-sota-evidence/v1';
export const PROPOSAL_SCHEMA = 'ruview.nightly-sota-proposal/v1';
export const BUNDLE_SCHEMA = 'ruview.nightly-sota-prototype/v1';
export const SCORE_SCHEMA = 'ruview.nightly-sota-score/v1';
export const VALIDATION_SCHEMA = 'ruview.nightly-sota-validation/v1';
export const RECEIPT_SCHEMA = 'ruview.nightly-sota-cognitum-receipt/v1';
export const TRANSFORM_SCHEMA = 'ruview.sota-transform/v1';
export const TEST_VECTORS_SCHEMA = 'ruview.sota-transform-tests/v1';
export const ISSUE_MARKER_PREFIX = '<!-- nightly-sota:fingerprint=';
export const EXPECTED_HARNESS_CHECK = 'Verify contributor harness';
export const GITHUB_ACTIONS_APP_ID = 15_368;
export const REGISTRY_URL = 'https://storage.googleapis.com/cognitum-apps/app-registry.json';
export const ARXIV_URL = new URL('https://export.arxiv.org/api/query');
ARXIV_URL.searchParams.set(
  'search_query',
  'all:"wifi sensing" OR all:"channel state information" OR all:"rf sensing" OR all:"wireless human pose"',
);
ARXIV_URL.searchParams.set('start', '0');
ARXIV_URL.searchParams.set('max_results', '30');
ARXIV_URL.searchParams.set('sortBy', 'submittedDate');
ARXIV_URL.searchParams.set('sortOrder', 'descending');

const ALLOWED_SUBSYSTEMS = new Set([
  'rf-sensing',
  'signal-processing',
  'edge-runtime',
  'simulation',
  'developer-tooling',
  'documentation',
  'homecore',
  'voice',
  'other',
]);
const ALLOWED_FINDING_CLASSES = new Set([
  'algorithm-evaluation',
  'benchmark-design',
  'documentation-gap',
  'integration-study',
  'representation-study',
  'robustness-study',
  'simulation-study',
  'tooling-study',
]);
const HIGH_RISK_RE =
  /\b(auth(?:entication|orization)?|credential|secret|crypt(?:o|ography)?|security|workflow|github actions|release|deploy(?:ment)?|production|firmware|hardware|bootloader|ota|dependency|dependencies|package manager|homecore|home assistant|voice|homekit pairing|mdns|network(?:ing)?|network server|websocket|rest(?:ful)? api|https? server|socket(?: listener)?|wasmtime|native plugin(?: api)?|plugin load(?:ing)?|ffi|unsafe|stt|tts|satellite voice)\b/i;
const MODEL_MARKUP_RE = /(?:https?:\/\/|www\.|<\/?[a-z][^>\r\n]{0,200}>|\[[^\]\r\n]{1,200}\]\([^)]+\)|```)/i;
const SECRET_RE = [
  /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/,
  /\b(?:cog_|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{12,}\b/,
  /\bAKIA[0-9A-Z]{16}\b/,
  /\b(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['"][^'"]{8,}['"]/i,
];

export function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

export function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function sorted(value) {
  if (Array.isArray(value)) return value.map(sorted);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sorted(value[key])]));
  }
  return value;
}

export function canonicalJson(value) {
  return JSON.stringify(sorted(value));
}

function exactKeys(value, keys, name) {
  invariant(value && typeof value === 'object' && !Array.isArray(value), `${name} must be an object`);
  invariant(
    canonicalJson(Object.keys(value).sort()) === canonicalJson([...keys].sort()),
    `${name} contains missing or unexpected keys`,
  );
}

export function cleanText(value, name, { min = 1, max = 2_000, singleLine = false } = {}) {
  invariant(typeof value === 'string', `${name} must be a string`);
  const text = value.replace(/\r\n?/g, '\n').trim();
  invariant(text.length >= min && text.length <= max, `${name} must contain ${min}-${max} characters`);
  invariant(!/[\0\u0001-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(text), `${name} contains control characters`);
  invariant(!/[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]/i.test(text), `${name} contains invisible or bidirectional controls`);
  invariant(!singleLine || !text.includes('\n'), `${name} must be one line`);
  return text;
}

export function cleanStringArray(value, name, { min = 0, max = 8, itemMax = 1_000 } = {}) {
  invariant(Array.isArray(value), `${name} must be an array`);
  invariant(value.length >= min && value.length <= max, `${name} must contain ${min}-${max} items`);
  const unique = [...new Set(value.map((item, index) => cleanText(item, `${name}[${index}]`, { max: itemMax })))];
  invariant(unique.length >= min && unique.length <= max, `${name} must contain ${min}-${max} unique items`);
  return unique;
}

export function escapeMarkdown(value) {
  return String(value)
    .replaceAll('\\', '\\\\')
    .replace(/([`*_[\]{}()#+.!|<>~-])/g, '\\$1')
    .replaceAll(':', '&#58;')
    .replaceAll('@', '&#64;');
}

export function safeGitHubTitle(value) {
  return cleanText(value, 'GitHub title', { max: 120, singleLine: true })
    .replaceAll('@', '[at]')
    .replaceAll('#', 'number ')
    .replace(/[<>{}[\]`]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

export async function readJson(file, maxBytes = 1_048_576) {
  const details = await stat(file);
  invariant(details.isFile(), `${file} is not a file`);
  invariant(details.size <= maxBytes, `${file} exceeds ${maxBytes} bytes`);
  const text = await readFile(file, 'utf8');
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`${file} is not valid JSON`);
  }
}

export async function writeJson(file, value) {
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, { encoding: 'utf8', flag: 'wx' });
}

export function parseArgs(argv) {
  const result = { _: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) {
      result._.push(token);
      continue;
    }
    const name = token.slice(2);
    invariant(/^[a-z][a-z0-9-]*$/.test(name), `invalid option: ${token}`);
    invariant(index + 1 < argv.length && !argv[index + 1].startsWith('--'), `missing value for ${token}`);
    invariant(result[name] === undefined, `duplicate option: ${token}`);
    result[name] = argv[index + 1];
    index += 1;
  }
  return result;
}

function allowedRemote(url, method) {
  if (url.protocol !== 'https:') return false;
  if (url.hostname === 'storage.googleapis.com') {
    return method === 'GET' && url.pathname === '/cognitum-apps/app-registry.json' && !url.search;
  }
  if (url.hostname === 'export.arxiv.org') return method === 'GET' && url.pathname === '/api/query';
  if (url.hostname === 'api.cognitum.one') {
    return method === 'POST' && url.pathname === '/v1/chat/completions' && !url.search;
  }
  return false;
}

export async function fetchBounded(
  input,
  {
    method = 'GET',
    headers = {},
    body,
    maxBytes = 1_048_576,
    timeoutMs = 30_000,
    mediaTypes = [],
  } = {},
) {
  const url = new URL(input);
  invariant(['GET', 'POST'].includes(method), `unsupported HTTP method: ${method}`);
  invariant(allowedRemote(url, method), `remote URL or method is not allowlisted: ${method} ${url.origin}${url.pathname}`);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(url, {
      method,
      headers,
      body,
      redirect: 'manual',
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timeout);
    if (error?.name === 'AbortError') throw new Error(`request timed out: ${url.hostname}`);
    throw new Error(`request failed: ${url.hostname}`);
  }
  try {
    invariant(response.status < 300 || response.status >= 400, `redirect rejected from ${url.hostname}`);
    invariant(response.ok, `request returned HTTP ${response.status} from ${url.hostname}`);
    const mediaType = (response.headers.get('content-type') || '').split(';')[0].trim().toLowerCase();
    if (mediaTypes.length) invariant(mediaTypes.includes(mediaType), `unexpected media type from ${url.hostname}: ${mediaType || 'missing'}`);
    const declared = Number(response.headers.get('content-length'));
    if (Number.isFinite(declared)) invariant(declared <= maxBytes, `response from ${url.hostname} exceeds ${maxBytes} bytes`);
    invariant(response.body, `empty response from ${url.hostname}`);
    const reader = response.body.getReader();
    const chunks = [];
    let size = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      invariant(size <= maxBytes, `response from ${url.hostname} exceeds ${maxBytes} bytes`);
      chunks.push(value);
    }
    const bytes = Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)), size);
    return { bytes, mediaType, requestId: response.headers.get('x-request-id') || null };
  } finally {
    clearTimeout(timeout);
  }
}

function decodeXml(value) {
  return value
    .replace(/<[^>]+>/g, ' ')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&')
    .replace(/\s+/g, ' ')
    .trim();
}

function xmlTag(block, tag) {
  const match = block.match(new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${tag}>`, 'i'));
  return match ? decodeXml(match[1]) : '';
}

export function parseArxivAtom(xml, now = new Date()) {
  invariant(typeof xml === 'string' && Buffer.byteLength(xml) <= 1_048_576, 'arXiv response is invalid or oversized');
  const records = [];
  const cutoff = now.getTime() - 370 * 24 * 60 * 60 * 1_000;
  for (const match of xml.matchAll(/<entry(?:\s[^>]*)?>([\s\S]*?)<\/entry>/gi)) {
    if (records.length >= 24) break;
    const entry = match[1];
    const rawId = xmlTag(entry, 'id');
    const title = xmlTag(entry, 'title');
    const summary = xmlTag(entry, 'summary');
    const published = xmlTag(entry, 'published');
    const publishedMs = Date.parse(published);
    if (!rawId || !title || !summary || !Number.isFinite(publishedMs) || publishedMs < cutoff) continue;
    let sourceUrl;
    try {
      const candidate = new URL(rawId.replace(/^http:/, 'https:'));
      if (candidate.hostname !== 'arxiv.org' || !candidate.pathname.startsWith('/abs/')) continue;
      sourceUrl = candidate.toString();
    } catch {
      continue;
    }
    const arxivId = sourceUrl.split('/abs/')[1].replace(/v\d+$/i, '');
    if (!/^[A-Za-z0-9./-]{4,80}$/.test(arxivId)) continue;
    const authors = [...entry.matchAll(/<author(?:\s[^>]*)?>([\s\S]*?)<\/author>/gi)]
      .slice(0, 12)
      .map((author) => xmlTag(author[1], 'name'))
      .filter(Boolean);
    records.push({
      id: `arxiv:${arxivId}`,
      kind: 'paper',
      classification: 'CLAIMED',
      title: title.slice(0, 300),
      summary: summary.slice(0, 2_500),
      url: sourceUrl,
      published: new Date(publishedMs).toISOString(),
      authors,
    });
  }
  return records;
}

export function normalizeCognitumRegistry(registry) {
  invariant(registry && typeof registry === 'object' && Array.isArray(registry.cogs), 'Cognitum registry schema mismatch');
  const version = cleanText(registry.version, 'registry.version', { max: 40, singleLine: true });
  const relevant = new Set(['research', 'signal', 'ai', 'developer', 'presence']);
  return registry.cogs
    .filter((cog) => cog && typeof cog === 'object' && relevant.has(cog.category))
    .slice(0, 60)
    .map((cog) => {
      const id = cleanText(cog.id, 'cog.id', { max: 80, singleLine: true });
      invariant(/^[a-z0-9][a-z0-9-]*$/.test(id), `invalid Cognitum cog id: ${id}`);
      return {
        id: `cognitum-cog:${id}:${cleanText(cog.version, 'cog.version', { max: 40, singleLine: true })}`,
        kind: 'cognitum-cog',
        classification: 'CLAIMED',
        title: cleanText(cog.name, 'cog.name', { max: 160, singleLine: true }),
        summary: cleanText(cog.description, 'cog.description', { max: 600 }),
        url: REGISTRY_URL,
        category: cog.category,
        registry_version: version,
      };
    });
}

export function validateEvidence(evidence, now = new Date()) {
  exactKeys(evidence, ['schema', 'collected_at', 'policy', 'query', 'snapshots', 'records'], 'evidence');
  invariant(evidence.schema === EVIDENCE_SCHEMA, `expected ${EVIDENCE_SCHEMA}`);
  const collectedAt = Date.parse(evidence.collected_at);
  invariant(Number.isFinite(collectedAt), 'evidence.collected_at is invalid');
  invariant(collectedAt <= now.getTime() + 5 * 60_000, 'evidence collection time is in the future');
  invariant(collectedAt >= now.getTime() - 2 * 24 * 60 * 60_000, 'evidence is older than two days');
  exactKeys(evidence.policy, ['untrusted', 'classifications', 'instruction_authority', 'max_age_days'], 'evidence.policy');
  invariant(evidence.policy.untrusted === true, 'evidence must be explicitly untrusted');
  invariant(evidence.policy.instruction_authority === false, 'evidence cannot have instruction authority');
  invariant(canonicalJson(evidence.policy.classifications) === canonicalJson(['CLAIMED']), 'evidence classifications changed');
  invariant(evidence.policy.max_age_days === 370, 'evidence age policy changed');
  exactKeys(evidence.query, ['arxiv', 'cognitum_categories'], 'evidence.query');
  invariant(evidence.query.arxiv === ARXIV_URL.searchParams.get('search_query'), 'arXiv query changed');
  invariant(
    canonicalJson(evidence.query.cognitum_categories) === canonicalJson(['research', 'signal', 'ai', 'developer', 'presence']),
    'Cognitum category query changed',
  );
  invariant(Array.isArray(evidence.snapshots) && evidence.snapshots.length === 2, 'evidence must bind two source snapshots');
  const expectedSnapshots = [
    { url: REGISTRY_URL, media_type: 'application/json' },
    { url: ARXIV_URL.toString(), media_type: 'application/atom+xml' },
  ];
  evidence.snapshots.forEach((snapshot, index) => {
    exactKeys(snapshot, ['url', 'media_type', 'bytes', 'sha256'], `snapshots[${index}]`);
    invariant(snapshot.url === expectedSnapshots[index].url, `snapshot ${index} URL changed`);
    invariant(snapshot.media_type === expectedSnapshots[index].media_type, `snapshot ${index} media type changed`);
    invariant(/^[a-f0-9]{64}$/.test(snapshot.sha256), 'snapshot digest is invalid');
    invariant(Number.isInteger(snapshot.bytes) && snapshot.bytes > 0 && snapshot.bytes <= 1_048_576, 'snapshot size is invalid');
  });
  invariant(Array.isArray(evidence.records) && evidence.records.length >= 2 && evidence.records.length <= 84, 'evidence record count is invalid');
  const ids = new Set();
  for (const [index, record] of evidence.records.entries()) {
    if (record?.kind === 'paper') {
      exactKeys(record, ['id', 'kind', 'classification', 'title', 'summary', 'url', 'published', 'authors'], `records[${index}]`);
    } else if (record?.kind === 'cognitum-cog') {
      exactKeys(record, ['id', 'kind', 'classification', 'title', 'summary', 'url', 'category', 'registry_version'], `records[${index}]`);
    } else {
      throw new Error(`unsupported evidence kind: ${record?.kind}`);
    }
    const id = cleanText(record.id, `records[${index}].id`, { max: 160, singleLine: true });
    invariant(!ids.has(id), `duplicate evidence id: ${id}`);
    ids.add(id);
    invariant(record.classification === 'CLAIMED', 'retrieved evidence must be labelled CLAIMED');
    cleanText(record.title, `records[${index}].title`, { max: 300 });
    cleanText(record.summary, `records[${index}].summary`, { max: 2_500 });
    if (record.kind === 'paper') {
      invariant(/^arxiv:[A-Za-z0-9./-]{4,80}$/.test(id), `invalid arXiv evidence id: ${id}`);
      const source = new URL(record.url);
      invariant(source.protocol === 'https:' && source.hostname === 'arxiv.org' && source.pathname.startsWith('/abs/'), 'paper URL is not canonical arXiv HTTPS');
      invariant(!source.search && !source.hash, 'paper URL cannot contain query or fragment');
      invariant(source.pathname.slice('/abs/'.length).replace(/v\d+$/i, '') === id.slice('arxiv:'.length), 'paper URL does not bind its evidence id');
      const published = Date.parse(record.published);
      invariant(Number.isFinite(published), 'paper publication date is invalid');
      invariant(published <= collectedAt + 24 * 60 * 60_000, 'paper publication date is in the future');
      invariant(published >= collectedAt - 370 * 24 * 60 * 60_000, 'paper is outside the evidence age window');
      const authors = cleanStringArray(record.authors, `records[${index}].authors`, { min: 1, max: 20, itemMax: 160 });
      invariant(canonicalJson(authors) === canonicalJson(record.authors), 'paper authors are not canonical');
    } else {
      invariant(/^cognitum-cog:[a-z0-9][a-z0-9-]*:[A-Za-z0-9._-]{1,40}$/.test(id), `invalid Cognitum evidence id: ${id}`);
      invariant(record.url === REGISTRY_URL, 'Cognitum record URL changed');
      invariant(['research', 'signal', 'ai', 'developer', 'presence'].includes(record.category), 'Cognitum category is outside the query');
      cleanText(record.registry_version, `records[${index}].registry_version`, { max: 40, singleLine: true });
    }
  }
  invariant([...ids].some((id) => id.startsWith('arxiv:')), 'evidence contains no paper records');
  invariant([...ids].some((id) => id.startsWith('cognitum-cog:')), 'evidence contains no Cognitum records');
  invariant(canonicalJson([...ids]) === canonicalJson([...ids].sort()), 'evidence records are not canonically sorted');
  return evidence;
}

function sourceCitation(record) {
  return {
    id: record.id,
    title: record.title,
    url: record.url,
    classification: 'CLAIMED',
  };
}

export function proposalFingerprint({ source_ids: sourceIds, finding_class: findingClass, subsystem }) {
  return sha256(canonicalJson({
    finding_class: findingClass,
    source_ids: [...sourceIds].sort(),
    subsystem,
  }));
}

export function classifyRisk(proposal) {
  return HIGH_RISK_RE.test(canonicalJson(proposal)) ? 'high' : 'low';
}

export function normalizeProposal(raw, evidence) {
  validateEvidence(evidence);
  invariant(raw && typeof raw === 'object' && !Array.isArray(raw), 'proposal must be an object');
  const modelKeys = ['finding_class', 'hypothesis', 'limitations', 'source_ids', 'subsystem', 'summary', 'title', 'unverified_claims', 'validation'];
  const canonicalKeys = [...modelKeys, 'citations', 'fingerprint', 'implementation', 'risk', 'schema'];
  const allowedKeys = raw.schema === PROPOSAL_SCHEMA ? canonicalKeys : modelKeys;
  invariant(
    canonicalJson(Object.keys(raw).sort()) === canonicalJson([...allowedKeys].sort()),
    'proposal contains missing or unexpected keys',
  );
  const records = new Map(evidence.records.map((record) => [record.id, record]));
  const sourceIds = cleanStringArray(raw.source_ids, 'source_ids', { min: 2, max: 8, itemMax: 160 });
  for (const id of sourceIds) invariant(records.has(id), `proposal cites unknown evidence id: ${id}`);
  const subsystem = cleanText(raw.subsystem, 'subsystem', { max: 60, singleLine: true }).toLowerCase();
  invariant(ALLOWED_SUBSYSTEMS.has(subsystem), `unsupported subsystem: ${subsystem}`);
  const findingClass = cleanText(raw.finding_class, 'finding_class', { max: 64, singleLine: true }).toLowerCase();
  invariant(ALLOWED_FINDING_CLASSES.has(findingClass), `unsupported finding_class: ${findingClass}`);
  const proposal = {
    schema: PROPOSAL_SCHEMA,
    title: cleanText(raw.title, 'title', { max: 120, singleLine: true }),
    summary: cleanText(raw.summary, 'summary', { max: 2_000 }),
    subsystem,
    finding_class: findingClass,
    hypothesis: cleanText(raw.hypothesis, 'hypothesis', { min: 40, max: 2_000 }),
    source_ids: [...sourceIds].sort(),
    citations: [...sourceIds].sort().map((id) => sourceCitation(records.get(id))),
    validation: cleanStringArray(raw.validation, 'validation', { min: 2, max: 8, itemMax: 500 }),
    limitations: cleanStringArray(raw.limitations, 'limitations', { min: 1, max: 8, itemMax: 500 }),
    unverified_claims: cleanStringArray(raw.unverified_claims, 'unverified_claims', { min: 1, max: 8, itemMax: 500 }),
  };
  proposal.risk = classifyRisk(proposal);
  proposal.implementation = {
    kind: proposal.risk === 'low' ? 'offline-prototype' : 'issue-only',
    target_root: proposal.risk === 'low' ? null : 'none',
  };
  proposal.fingerprint = proposalFingerprint(proposal);
  if (proposal.risk === 'low') {
    proposal.implementation.target_root = `examples/research-sota/nightly/${proposal.fingerprint.slice(0, 16)}`;
  }
  return proposal;
}

export function validateProposal(proposal, evidence) {
  invariant(proposal?.schema === PROPOSAL_SCHEMA, `expected ${PROPOSAL_SCHEMA}`);
  const normalized = normalizeProposal(proposal, evidence);
  invariant(canonicalJson(normalized) === canonicalJson(proposal), 'proposal is not canonical or has locally governed fields changed');
  return proposal;
}

export function scoreProposal(proposal) {
  invariant(proposal?.schema === PROPOSAL_SCHEMA, `expected ${PROPOSAL_SCHEMA}`);
  const checks = {
    has_multiple_sources: proposal.source_ids.length >= 2,
    has_primary_research: proposal.citations.some((item) => item.id.startsWith('arxiv:')),
    maps_cognitum_surface: proposal.citations.some((item) => item.id.startsWith('cognitum-cog:')),
    states_testable_hypothesis: proposal.hypothesis.length >= 80,
    names_multiple_validations: proposal.validation.length >= 2,
    records_limitations: proposal.limitations.length >= 1,
    records_unverified_claims: proposal.unverified_claims.length >= 1,
    names_local_risk: ['low', 'high'].includes(proposal.risk),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  return {
    schema: SCORE_SCHEMA,
    score_name: 'PROPOSAL_COMPLETENESS',
    score: passed / Object.keys(checks).length,
    checks,
    disclaimer: 'This deterministic completeness score is not a scientific-quality, novelty, safety, or performance claim.',
  };
}

export function extractModelText(response) {
  invariant(response && typeof response === 'object', 'Cognitum returned a non-object response');
  const chat = response.choices?.[0]?.message?.content;
  if (typeof chat === 'string' && chat.trim()) return chat.trim();
  if (Array.isArray(chat)) {
    const joined = chat.map((part) => (typeof part === 'string' ? part : part?.text || '')).join('');
    if (joined.trim()) return joined.trim();
  }
  if (typeof response.output_text === 'string' && response.output_text.trim()) return response.output_text.trim();
  if (Array.isArray(response.output)) {
    const joined = response.output
      .flatMap((item) => item?.content || [])
      .map((part) => part?.text || part?.output_text || '')
      .join('');
    if (joined.trim()) return joined.trim();
  }
  throw new Error('Cognitum response did not contain model text');
}

export function parseModelJson(text, maxBytes = 524_288) {
  invariant(typeof text === 'string' && Buffer.byteLength(text) <= maxBytes, 'model output is invalid or oversized');
  const candidate = text.trim();
  invariant(candidate.startsWith('{') && candidate.endsWith('}'), 'model output must be one JSON object');
  try {
    return JSON.parse(candidate);
  } catch {
    throw new Error('model output is not valid JSON');
  }
}

export function validateCognitumReceipt(receipt, normalizedOutputSha256, expectedRequestSha256) {
  exactKeys(
    receipt,
    [
      'schema',
      'provider',
      'endpoint',
      'requested_model',
      'response_model',
      'request_id',
      'request_sha256',
      'raw_output_sha256',
      'normalized_output_sha256',
      'routing',
      'routing_attestation_sha256',
    ],
    'Cognitum receipt',
  );
  invariant(receipt.schema === RECEIPT_SCHEMA, `expected ${RECEIPT_SCHEMA}`);
  invariant(receipt.provider === 'cognitum' && receipt.endpoint === '/v1/chat/completions', 'Cognitum receipt endpoint changed');
  invariant(receipt.requested_model === 'cognitum-mid', 'Cognitum requested model changed');
  invariant(receipt.response_model === 'cognitum-mid', 'Cognitum response model changed');
  invariant(
    typeof receipt.request_id === 'string' && /^[A-Za-z0-9._:-]{1,160}$/.test(receipt.request_id),
    'Cognitum request id is invalid',
  );
  invariant(
    typeof expectedRequestSha256 === 'string' &&
      /^[a-f0-9]{64}$/.test(expectedRequestSha256) &&
      receipt.request_sha256 === expectedRequestSha256,
    'Cognitum request digest mismatch',
  );
  invariant(/^[a-f0-9]{64}$/.test(receipt.raw_output_sha256), 'Cognitum output digest is invalid');
  invariant(receipt.normalized_output_sha256 === normalizedOutputSha256, 'Cognitum receipt does not bind normalized output');
  exactKeys(
    receipt.routing,
    ['request_id', 'resolved_tier', 'resolved_model', 'escalated', 'cap_degraded'],
    'Cognitum routing attestation',
  );
  invariant(receipt.routing.request_id === receipt.request_id, 'Cognitum routing request id mismatch');
  invariant(
    receipt.routing.resolved_tier === 'mid' && receipt.routing.resolved_model === 'cognitum-mid',
    'Cognitum route did not resolve to cognitum-mid',
  );
  invariant(
    receipt.routing.escalated === false && receipt.routing.cap_degraded === false,
    'Cognitum route was escalated or degraded',
  );
  invariant(
    receipt.routing_attestation_sha256 === sha256(canonicalJson(receipt.routing)),
    'Cognitum routing attestation digest mismatch',
  );
  return receipt;
}

export function evaluateMainProtection(effectiveRules) {
  invariant(Array.isArray(effectiveRules), 'GitHub branch rules response is not an array');
  const reviewRule = effectiveRules.find(
    (rule) =>
      rule?.type === 'pull_request' &&
      Number.isInteger(rule.parameters?.required_approving_review_count) &&
      rule.parameters.required_approving_review_count >= 1,
  );
  const checksRule = effectiveRules.find(
    (rule) =>
      rule?.type === 'required_status_checks' &&
      Array.isArray(rule.parameters?.required_status_checks) &&
      rule.parameters.required_status_checks.some(
        (check) =>
          check?.context === EXPECTED_HARNESS_CHECK &&
          check?.integration_id === GITHUB_ACTIONS_APP_ID,
      ),
  );
  return {
    ready: Boolean(reviewRule && checksRule),
    reviewRule: Boolean(reviewRule),
    checksRule: Boolean(checksRule),
  };
}

export function validateHonestNullReplay(replay) {
  invariant(replay && typeof replay === 'object' && !Array.isArray(replay), 'Flywheel replay is invalid');
  invariant(replay.data_source === 'SYNTHETIC' && replay.root_id === 'ruview-gen0', 'Flywheel replay root changed');
  invariant(
    Array.isArray(replay.chain) && replay.chain.length === 1 && replay.chain[0]?.verdict === 'ROOT',
    'Flywheel promoted chain is not root-only',
  );
  invariant(Array.isArray(replay.all_commits) && replay.all_commits.length >= 1, 'Flywheel replay has no candidate evidence');
  invariant(
    replay.all_commits.every((commit) => commit?.verdict === 'REJECTED'),
    'Flywheel replay contains a non-rejected candidate',
  );
  invariant(
    replay.verified_improvements === 0 && replay.anchor_surviving_improvements === 0,
    'Flywheel replay is not honest-null',
  );
  invariant(replay.milestone_reached === false, 'Flywheel replay claims a milestone');
  return replay;
}

function boundedNumber(value, name, min = -1_000_000, max = 1_000_000) {
  invariant(typeof value === 'number' && Number.isFinite(value), `${name} must be a finite number`);
  invariant(value >= min && value <= max, `${name} is outside ${min}..${max}`);
  return Object.is(value, -0) ? 0 : value;
}

function normalizeOperation(operation, index) {
  invariant(operation && typeof operation === 'object' && !Array.isArray(operation), `pipeline[${index}] must be an object`);
  const name = operation.op;
  if (['center', 'normalize-peak', 'absolute', 'square'].includes(name)) {
    exactKeys(operation, ['op'], `pipeline[${index}]`);
    return { op: name };
  }
  if (name === 'difference') {
    exactKeys(operation, ['op', 'lag'], `pipeline[${index}]`);
    invariant(Number.isInteger(operation.lag) && operation.lag >= 1 && operation.lag <= 16, `pipeline[${index}].lag is invalid`);
    return { op: name, lag: operation.lag };
  }
  if (name === 'moving-average') {
    exactKeys(operation, ['op', 'window'], `pipeline[${index}]`);
    invariant(Number.isInteger(operation.window) && operation.window >= 2 && operation.window <= 32, `pipeline[${index}].window is invalid`);
    return { op: name, window: operation.window };
  }
  if (name === 'clip') {
    exactKeys(operation, ['op', 'min', 'max'], `pipeline[${index}]`);
    const min = boundedNumber(operation.min, `pipeline[${index}].min`);
    const max = boundedNumber(operation.max, `pipeline[${index}].max`);
    invariant(min < max, `pipeline[${index}] clip bounds are reversed`);
    return { op: name, min, max };
  }
  throw new Error(`pipeline[${index}] has unsupported operation: ${name}`);
}

export function normalizeTransformSpec(value) {
  exactKeys(value, ['schema', 'name', 'description', 'input_kind', 'pipeline'], 'prototype');
  invariant(value.schema === TRANSFORM_SCHEMA, `expected ${TRANSFORM_SCHEMA}`);
  const name = cleanText(value.name, 'prototype.name', { max: 64, singleLine: true }).toLowerCase();
  invariant(/^[a-z0-9][a-z0-9-]{1,63}$/.test(name), 'prototype.name must be a lowercase slug');
  const description = cleanText(value.description, 'prototype.description', { min: 20, max: 500 });
  invariant(value.input_kind === 'scalar-series', 'prototype.input_kind must be scalar-series');
  invariant(Array.isArray(value.pipeline) && value.pipeline.length >= 1 && value.pipeline.length <= 8, 'prototype.pipeline must contain 1-8 operations');
  return {
    schema: TRANSFORM_SCHEMA,
    name,
    description,
    input_kind: 'scalar-series',
    pipeline: value.pipeline.map(normalizeOperation),
  };
}

export function evaluateTransform(spec, input) {
  const normalized = normalizeTransformSpec(spec);
  invariant(Array.isArray(input) && input.length >= 2 && input.length <= 128, 'transform input must contain 2-128 values');
  let values = input.map((value, index) => boundedNumber(value, `input[${index}]`));
  for (const operation of normalized.pipeline) {
    if (operation.op === 'center') {
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      values = values.map((value) => value - mean);
    } else if (operation.op === 'normalize-peak') {
      const peak = Math.max(...values.map(Math.abs));
      values = peak === 0 ? values.map(() => 0) : values.map((value) => value / peak);
    } else if (operation.op === 'absolute') {
      values = values.map(Math.abs);
    } else if (operation.op === 'square') {
      values = values.map((value) => value * value);
    } else if (operation.op === 'difference') {
      invariant(values.length > operation.lag, `difference lag ${operation.lag} empties the series`);
      values = values.slice(operation.lag).map((value, index) => value - values[index]);
    } else if (operation.op === 'moving-average') {
      invariant(values.length >= operation.window, `moving-average window ${operation.window} exceeds the series`);
      values = Array.from(
        { length: values.length - operation.window + 1 },
        (_, index) => values.slice(index, index + operation.window).reduce((sum, value) => sum + value, 0) / operation.window,
      );
    } else if (operation.op === 'clip') {
      values = values.map((value) => Math.max(operation.min, Math.min(operation.max, value)));
    }
    invariant(values.every((value) => Number.isFinite(value) && Math.abs(value) <= 1_000_000_000_000), 'transform produced an unbounded value');
  }
  return values.map((value) => (Object.is(value, -0) ? 0 : value));
}

export function normalizeTestVectors(value, spec) {
  exactKeys(value, ['schema', 'cases'], 'test_vectors');
  invariant(value.schema === TEST_VECTORS_SCHEMA, `expected ${TEST_VECTORS_SCHEMA}`);
  invariant(Array.isArray(value.cases) && value.cases.length >= 2 && value.cases.length <= 8, 'test_vectors.cases must contain 2-8 cases');
  const names = new Set();
  const cases = value.cases.map((item, index) => {
    exactKeys(item, ['name', 'input', 'expected'], `test_vectors.cases[${index}]`);
    const name = cleanText(item.name, `test_vectors.cases[${index}].name`, { max: 64, singleLine: true }).toLowerCase();
    invariant(/^[a-z0-9][a-z0-9-]{1,63}$/.test(name) && !names.has(name), `test vector name is invalid or duplicated: ${name}`);
    names.add(name);
    invariant(Array.isArray(item.input), `test vector ${name} input must be an array`);
    invariant(Array.isArray(item.expected), `test vector ${name} expected must be an array`);
    invariant(item.input.length >= 2 && item.input.length <= 128, `test vector ${name} input length is invalid`);
    invariant(item.expected.length >= 1 && item.expected.length <= 128, `test vector ${name} expected length is invalid`);
    const input = item.input.map((number, itemIndex) => boundedNumber(number, `${name}.input[${itemIndex}]`));
    const expected = item.expected.map((number, itemIndex) => boundedNumber(number, `${name}.expected[${itemIndex}]`, -1_000_000_000_000, 1_000_000_000_000));
    const actual = evaluateTransform(spec, input);
    invariant(actual.length === expected.length, `test vector ${name} output length mismatch`);
    invariant(
      actual.every((number, itemIndex) => Math.abs(number - expected[itemIndex]) <= 1e-9 * (1 + Math.abs(expected[itemIndex]))),
      `test vector ${name} expected values do not match the governed transform`,
    );
    return { name, input, expected };
  });
  return { schema: TEST_VECTORS_SCHEMA, cases };
}

function jsonFile(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function renderPrototypeSource(spec) {
  return `// Generated from a schema-checked declarative transform. Do not hand-edit.
const PIPELINE = Object.freeze(${JSON.stringify(spec.pipeline, null, 2)});

function finiteSeries(input) {
  if (!Array.isArray(input) || input.length < 2 || input.length > 128) {
    throw new TypeError('input must contain 2-128 finite numbers');
  }
  if (!input.every((value) => typeof value === 'number' && Number.isFinite(value))) {
    throw new TypeError('input contains a non-finite number');
  }
  const values = [...input];
  if (!values.every((value) => Math.abs(value) <= 1_000_000)) {
    throw new RangeError('input value exceeds the governed bound');
  }
  return values;
}

export function run(input) {
  let values = finiteSeries(input);
  for (const operation of PIPELINE) {
    if (operation.op === 'center') {
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      values = values.map((value) => value - mean);
    } else if (operation.op === 'normalize-peak') {
      const peak = Math.max(...values.map(Math.abs));
      values = peak === 0 ? values.map(() => 0) : values.map((value) => value / peak);
    } else if (operation.op === 'absolute') {
      values = values.map(Math.abs);
    } else if (operation.op === 'square') {
      values = values.map((value) => value * value);
    } else if (operation.op === 'difference') {
      if (values.length <= operation.lag) throw new RangeError('difference lag empties the series');
      values = values.slice(operation.lag).map((value, index) => value - values[index]);
    } else if (operation.op === 'moving-average') {
      if (values.length < operation.window) throw new RangeError('moving-average window exceeds the series');
      values = Array.from(
        { length: values.length - operation.window + 1 },
        (_, index) => values.slice(index, index + operation.window)
          .reduce((sum, value) => sum + value, 0) / operation.window,
      );
    } else if (operation.op === 'clip') {
      values = values.map((value) => Math.max(operation.min, Math.min(operation.max, value)));
    }
    if (!values.every((value) => Number.isFinite(value) && Math.abs(value) <= 1_000_000_000_000)) {
      throw new RangeError('transform output exceeds the governed bound');
    }
  }
  return values.map((value) => Object.is(value, -0) ? 0 : value);
}
`;
}

function renderPrototypeTests(vectors) {
  return `import test from 'node:test';
import assert from 'node:assert/strict';
import { run } from './prototype.mjs';

const CASES = ${JSON.stringify(vectors.cases, null, 2)};

for (const fixture of CASES) {
  test(fixture.name, () => {
    const actual = run(fixture.input);
    assert.equal(actual.length, fixture.expected.length);
    actual.forEach((value, index) => {
      const tolerance = 1e-9 * (1 + Math.abs(fixture.expected[index]));
      assert.ok(Math.abs(value - fixture.expected[index]) <= tolerance);
    });
  });
}
`;
}

function renderPrototypeReadme(proposal, summary, notes, spec) {
  const citations = proposal.citations
    .map((item) => `- [${escapeMarkdown(item.title)}](${item.url}) -- ${item.classification}`)
    .join('\n');
  return `# ${escapeMarkdown(proposal.title)}

> Automatically generated, declarative offline prototype. Research behavior is unvalidated and remains CLAIMED.

${escapeMarkdown(summary)}

## Hypothesis

${escapeMarkdown(proposal.hypothesis)}

## Governed transform

- Name: \`${escapeMarkdown(spec.name)}\`
- Input: scalar series
- Operations: ${spec.pipeline.map((operation) => `\`${escapeMarkdown(operation.op)}\``).join(', ')}

The model selected only schema-bounded data. \`prototype.mjs\` and
\`prototype.test.mjs\` were emitted by a trusted repository template. The
nightly workflow syntax-checks but does not execute generated files.

## Notes

${notes.map((note) => `- ${escapeMarkdown(note)}`).join('\n')}

## Source evidence

${citations}

## Limitations

${proposal.limitations.map((item) => `- ${escapeMarkdown(item)}`).join('\n')}
`;
}

export function normalizePrototypeBundle(raw, proposal) {
  invariant(proposal?.schema === PROPOSAL_SCHEMA, `expected ${PROPOSAL_SCHEMA}`);
  invariant(proposal.risk === 'low' && proposal.implementation.kind === 'offline-prototype', 'proposal is issue-only');
  exactKeys(raw, ['summary', 'notes', 'prototype', 'test_vectors'], 'prototype bundle');
  const summary = cleanText(raw.summary, 'bundle.summary', { max: 1_000 });
  const notes = cleanStringArray(raw.notes, 'bundle.notes', { min: 1, max: 6, itemMax: 400 });
  const spec = normalizeTransformSpec(raw.prototype);
  const vectors = normalizeTestVectors(raw.test_vectors, spec);
  invariant(
    !MODEL_MARKUP_RE.test(canonicalJson({ summary, notes, description: spec.description })),
    'prototype prose contains a URL, HTML, or fenced Markdown',
  );
  invariant(
    !HIGH_RISK_RE.test(canonicalJson({ summary, notes, spec, vectors })),
    'prototype introduced a high-risk topic after proposal triage',
  );
  for (const pattern of SECRET_RE) {
    invariant(!pattern.test(canonicalJson({ summary, notes, spec, vectors })), 'prototype contains a secret-shaped value');
  }
  const root = proposal.implementation.target_root;
  const files = [
    { path: `${root}/README.md`, content: renderPrototypeReadme(proposal, summary, notes, spec) },
    { path: `${root}/prototype.json`, content: jsonFile(spec) },
    { path: `${root}/prototype.mjs`, content: renderPrototypeSource(spec) },
    { path: `${root}/prototype.test.mjs`, content: renderPrototypeTests(vectors) },
    { path: `${root}/test-vectors.json`, content: jsonFile(vectors) },
  ].sort((a, b) => a.path.localeCompare(b.path));
  const totalBytes = files.reduce((sum, file) => sum + Buffer.byteLength(file.content), 0);
  const totalLines = files.reduce((sum, file) => sum + file.content.split('\n').length - 1, 0);
  invariant(totalBytes <= 262_144, 'prototype exceeds 256 KiB');
  invariant(totalLines <= 400, 'prototype exceeds 400 lines');
  return {
    schema: BUNDLE_SCHEMA,
    proposal_fingerprint: proposal.fingerprint,
    summary,
    notes,
    constraints: {
      new_files_only: true,
      offline_only: true,
      declarative_model_output_only: true,
      trusted_code_template: true,
      generated_code_executed: false,
      file_count: 5,
      max_lines: 400,
      max_bytes: 262_144,
    },
    files,
  };
}

export function validatePrototypeBundle(bundle, proposal) {
  invariant(bundle?.schema === BUNDLE_SCHEMA, `expected ${BUNDLE_SCHEMA}`);
  exactKeys(bundle, ['schema', 'proposal_fingerprint', 'summary', 'notes', 'constraints', 'files'], 'canonical prototype bundle');
  invariant(Array.isArray(bundle.files) && bundle.files.length === 5, 'canonical prototype bundle must contain five files');
  const prototypeFile = bundle.files.find((file) => file.path.endsWith('/prototype.json'));
  const vectorsFile = bundle.files.find((file) => file.path.endsWith('/test-vectors.json'));
  invariant(prototypeFile && vectorsFile, 'canonical prototype JSON files are missing');
  let prototype;
  let testVectors;
  try {
    prototype = JSON.parse(prototypeFile.content);
    testVectors = JSON.parse(vectorsFile.content);
  } catch {
    throw new Error('canonical prototype JSON is invalid');
  }
  const normalized = normalizePrototypeBundle({
    summary: bundle.summary,
    notes: bundle.notes,
    prototype,
    test_vectors: testVectors,
  }, proposal);
  invariant(canonicalJson(normalized) === canonicalJson(bundle), 'prototype bundle is not canonical or has governed fields changed');
  return bundle;
}

export function bundleDigest(bundle) {
  return sha256(canonicalJson(bundle));
}

export async function assertPrototypePathsAbsent(bundle, repoRoot) {
  const root = await realpath(path.resolve(repoRoot));
  for (const file of bundle.files) {
    const target = path.resolve(root, ...file.path.split('/'));
    invariant(target.startsWith(`${root}${path.sep}`), `prototype path escapes repository: ${file.path}`);
    let current = root;
    const components = file.path.split('/');
    for (const component of components.slice(0, -1)) {
      current = path.join(current, component);
      try {
        const details = await lstat(current);
        invariant(!details.isSymbolicLink(), `prototype parent is a symlink: ${file.path}`);
        invariant(details.isDirectory(), `prototype parent is not a directory: ${file.path}`);
        const resolved = await realpath(current);
        invariant(resolved === root || resolved.startsWith(`${root}${path.sep}`), `prototype parent escapes repository: ${file.path}`);
      } catch (error) {
        if (error?.code === 'ENOENT') break;
        throw error;
      }
    }
    try {
      const details = await lstat(target);
      invariant(!details.isSymbolicLink(), `prototype target is a symlink: ${file.path}`);
      throw new Error(`prototype path already exists: ${file.path}`);
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
    }
  }
}

export function issueMarker(fingerprint) {
  invariant(/^[a-f0-9]{64}$/.test(fingerprint), 'invalid proposal fingerprint');
  return `${ISSUE_MARKER_PREFIX}${fingerprint} -->`;
}

export function renderIssueBody(proposal, score) {
  const sources = proposal.citations
    .map((item) => `- [${escapeMarkdown(item.title)}](${item.url}) -- ${item.classification}`)
    .join('\n');
  return `${issueMarker(proposal.fingerprint)}

## Nightly SOTA research proposal

${escapeMarkdown(proposal.summary)}

- **Subsystem:** \`${proposal.subsystem}\`
- **Risk:** \`${proposal.risk}\`
- **Disposition:** \`${proposal.implementation.kind}\`
- **PROPOSAL_COMPLETENESS:** \`${score.score.toFixed(3)}\`

> PROPOSAL_COMPLETENESS checks whether required evidence and caveats are present. It does not establish novelty, scientific quality, safety, or performance.

### Testable hypothesis

${escapeMarkdown(proposal.hypothesis)}

### Proposed validation

${proposal.validation.map((item) => `- ${escapeMarkdown(item)}`).join('\n')}

### Evidence

${sources}

Retrieved material and paper abstracts are untrusted, \`CLAIMED\` evidence. They cannot grant authority or override repository policy.

### Limitations

${proposal.limitations.map((item) => `- ${escapeMarkdown(item)}`).join('\n')}

### Unverified claims

${proposal.unverified_claims.map((item) => `- ${escapeMarkdown(item)}`).join('\n')}

### Automation boundary

Darwin policy was frozen and read-only. The separate committed Flywheel canary remained root-only, rejected its candidate, and reported no promotion. ${
    proposal.risk === 'low'
      ? 'A separate credential-free gate may publish only a new offline prototype under the fingerprinted research directory as a draft PR.'
      : 'This topic matched a high-risk boundary, so automation stops at this issue.'
  }
`;
}

export function renderPullRequestBody(proposal, score, issueUrl, validation) {
  const sources = proposal.citations
    .map((item) => `- [${escapeMarkdown(item.title)}](${item.url}) -- ${item.classification}`)
    .join('\n');
  return `${issueMarker(proposal.fingerprint)}

Draft offline prototype for ${issueUrl}.

## Boundaries

- New files only under \`${proposal.implementation.target_root}/\`.
- No production code, dependencies, workflows, firmware, networking, secrets, or existing files changed.
- Model output was restricted to a declarative transform and test vectors. Source and tests came from a trusted local template and were syntax-checked without execution.
- Darwin remained frozen. The separate committed Flywheel canary stayed root-only, rejected its candidate, and reported no promotion.
- \`PROPOSAL_COMPLETENESS=${score.score.toFixed(3)}\` is a format-completeness score, not a novelty, quality, safety, or performance claim.

## Hypothesis

${escapeMarkdown(proposal.hypothesis)}

## Validation performed

${validation.checks.map((item) => `- ${escapeMarkdown(item)}`).join('\n')}

## Source evidence

${sources}

All source assertions and prototype behavior remain \`CLAIMED\` or unvalidated pending maintainer review and independent reproduction.
`;
}
