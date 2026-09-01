export interface NlosServerUrlValidation {
  valid: boolean;
  error?: string;
  normalized?: string;
}

// Hermes does not provide TextEncoder in every supported React Native build.
// Count UTF-8 bytes without allocating an encoded copy.
const utf8Length = (value: string): number => {
  let bytes = 0;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code < 0x80) bytes += 1;
    else if (code < 0x800) bytes += 2;
    else if (code >= 0xd800 && code <= 0xdbff && index + 1 < value.length) {
      const next = value.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        bytes += 4;
        index += 1;
      } else bytes += 3;
    } else bytes += 3;
  }
  return bytes;
};

const isLoopback = (hostname: string): boolean =>
  hostname === 'localhost' ||
  hostname === '127.0.0.1' ||
  hostname === '[::1]' ||
  hostname === '::1';

/** Validate and reduce an NLOS endpoint to an origin-only URL before storage. */
export const normalizeNlosServerUrl = (raw: string): NlosServerUrlValidation => {
  const value = raw.trim();
  if (!value || utf8Length(value) > 2_048) {
    return { valid: false, error: 'NLOS server URL must be 1 to 2048 bytes.' };
  }

  try {
    const url = new URL(value);
    const secure = url.protocol === 'https:';
    const loopbackDevelopment = url.protocol === 'http:' && isLoopback(url.hostname);
    if (!secure && !loopbackDevelopment) {
      return {
        valid: false,
        error: 'NLOS requires HTTPS, except for a loopback development server.',
      };
    }
    if (
      url.username ||
      url.password ||
      url.search ||
      url.hash ||
      (url.pathname !== '' && url.pathname !== '/')
    ) {
      return {
        valid: false,
        error: 'Store only the NLOS server origin; credentials, paths, queries, and fragments are forbidden.',
      };
    }
    return { valid: true, normalized: url.origin };
  } catch {
    return { valid: false, error: 'Enter a valid NLOS server origin.' };
  }
};
