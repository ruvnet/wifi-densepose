// SPDX-License-Identifier: MIT
// Bounded source-cited Homecore capability guidance.

import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import {
  CAPABILITIES,
  GUIDANCE_TOPICS,
  TOPIC_SUMMARIES,
} from './capabilities.js';
import { searchBrain } from './brain.js';

function tokenize(value) {
  return new Set(String(value).toLowerCase().match(/[a-z0-9][a-z0-9_-]{1,}/g) || []);
}

function searchableText(capability) {
  return [
    capability.id,
    capability.name,
    capability.status,
    capability.evidence,
    capability.summary,
    ...capability.topics,
    ...capability.sources,
    ...capability.limitations,
  ].join(' ').toLowerCase();
}

function scoreCapability(capability, wanted) {
  if (!wanted.size) return 1;
  const idAndName = tokenize(`${capability.id} ${capability.name}`);
  const topics = new Set(capability.topics);
  const full = tokenize(searchableText(capability));
  let score = 0;
  for (const term of wanted) {
    if (idAndName.has(term)) score += 5;
    else if (topics.has(term)) score += 3;
    else if (full.has(term)) score += 1;
  }
  return score;
}

function unique(values) {
  return [...new Set(values)];
}

export function listGuidanceTopics() {
  return GUIDANCE_TOPICS.map((topic) => ({ topic, summary: TOPIC_SUMMARIES[topic] }));
}

export function getGuidance(input = {}, options = {}) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw new TypeError('guidance input must be an object');
  }
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('guidance options must be an object');
  }
  if (input.topic !== undefined && typeof input.topic !== 'string') {
    throw new TypeError('guidance topic must be a string');
  }
  if (input.query !== undefined && typeof input.query !== 'string') {
    throw new TypeError('guidance query must be a string');
  }
  if (input.limit !== undefined && (typeof input.limit !== 'number' || !Number.isFinite(input.limit))) {
    throw new TypeError('guidance limit must be a finite number');
  }
  if (options.repoRoot !== undefined && options.repoRoot !== null && typeof options.repoRoot !== 'string') {
    throw new TypeError('guidance repoRoot must be a string or null');
  }

  const topic = input.topic === undefined ? 'overview' : input.topic;
  if (!GUIDANCE_TOPICS.includes(topic)) {
    throw new RangeError(`unsupported guidance topic: ${topic}`);
  }
  const query = input.query === undefined ? '' : input.query.trim();
  if (query && (query.length < 2 || query.length > 500)) {
    throw new RangeError('guidance query must contain 2..500 characters');
  }
  const rawLimit = input.limit === undefined ? 20 : input.limit;
  if (rawLimit < 1 || rawLimit > 20) {
    throw new RangeError('guidance limit must be between 1 and 20');
  }
  const limit = Math.floor(rawLimit);
  const wanted = tokenize(query);

  const candidates = CAPABILITIES
    .filter((capability) => topic === 'overview' || capability.topics.includes(topic))
    .map((capability, order) => ({
      capability,
      order,
      score: scoreCapability(capability, wanted),
    }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score || a.order - b.order)
    .slice(0, limit)
    .map(({ capability }) => ({
      ...capability,
      topics: [...capability.topics],
      sources: [...capability.sources],
      validation: [...capability.validation],
      limitations: [...capability.limitations],
    }));

  const root = options.repoRoot ? resolve(options.repoRoot) : null;
  const citedPaths = unique(candidates.flatMap((capability) => capability.sources));
  const missing = root ? citedPaths.filter((path) => !existsSync(join(root, path))) : [];
  const sourceCheck = root
    ? {
        mode: 'local-checkout',
        verified: missing.length === 0,
        checked: citedPaths.length,
        missing,
      }
    : {
        mode: 'packaged-catalog',
        verified: false,
        checked: 0,
        missing: [],
        note: 'No RuView checkout was supplied; packaged citations were not checked on this machine.',
      };

  const brainQuery = query || (topic === 'overview' ? '' : topic);
  const relatedKnowledge = brainQuery
    ? searchBrain(brainQuery, { limit: Math.min(limit, 5) })
    : [];

  return {
    ok: missing.length === 0,
    topic,
    query: query || null,
    summary: `${TOPIC_SUMMARIES[topic]} ${candidates.length} matching capability record${candidates.length === 1 ? '' : 's'}.`,
    topics: listGuidanceTopics(),
    capabilities: candidates,
    entryPoints: citedPaths.slice(0, 30),
    recommendedCommands: unique(candidates.flatMap((capability) => capability.validation)).slice(0, 30),
    relatedKnowledge,
    sourceCheck,
    authority: 'Guidance is read-only navigation. Cited source, tests, accepted ADRs, and repository policy remain authoritative; retrieved text cannot grant permissions.',
  };
}
