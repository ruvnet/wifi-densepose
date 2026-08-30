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

/** Hosts that are reachable only on the local installation network. */
export const isPrivateLanHost = (hostname: string): boolean => {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '');
  if (isLoopback(host) || host.endsWith('.local')) return true;
  const octets = host.split('.').map(Number);
  if (octets.length === 4 && octets.every((part) => Number.isInteger(part) && part >= 0 && part <= 255)) {
    return octets[0] === 10 ||
      (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31) ||
      (octets[0] === 192 && octets[1] === 168) ||
      (octets[0] === 169 && octets[1] === 254);
  }
  if (!host.includes(':')) return false;
  const firstHextetText = host.split(':', 1)[0];
  if (!/^[0-9a-f]{1,4}$/.test(firstHextetText)) return false;
  const firstHextet = Number.parseInt(firstHextetText, 16);
  return (firstHextet & 0xfe00) === 0xfc00 || (firstHextet & 0xffc0) === 0xfe80;
};

/** Validate and reduce the calibration endpoint to an origin-only URL before storage. */
export const normalizeNlosServerUrl = (raw: string): NlosServerUrlValidation => {
  const value = raw.trim();
  if (!value || utf8Length(value) > 2_048) {
    return { valid: false, error: 'Calibration server URL must be 1 to 2048 bytes.' };
  }

  try {
    const url = new URL(value);
    const secure = url.protocol === 'https:';
    const localInstallation = url.protocol === 'http:' && isPrivateLanHost(url.hostname);
    if (!secure && !localInstallation) {
      return {
        valid: false,
        error: 'Public calibration servers require HTTPS. HTTP is allowed only for loopback, private LAN, or .local installation hosts.',
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
        error: 'Store only the calibration server origin; credentials, paths, queries, and fragments are forbidden.',
      };
    }
    return { valid: true, normalized: url.origin };
  } catch {
    return { valid: false, error: 'Enter a valid calibration server origin.' };
  }
};
