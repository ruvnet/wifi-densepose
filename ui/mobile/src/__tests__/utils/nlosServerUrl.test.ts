import { isPrivateLanHost, normalizeNlosServerUrl } from '@/utils/nlosServerUrl';

describe('calibration server transport policy', () => {
  it.each(['192.168.1.166', '10.4.0.9', '172.20.1.2', '169.254.2.3', 'ruview.local', 'fd00::1'])('recognizes local installation host %s', (host) => {
    expect(isPrivateLanHost(host)).toBe(true);
  });

  it('accepts origin-only HTTP on the private LAN', () => {
    expect(normalizeNlosServerUrl('http://192.168.1.166:3000')).toEqual({ valid: true, normalized: 'http://192.168.1.166:3000' });
  });

  it('continues to require HTTPS for public authorities', () => {
    expect(normalizeNlosServerUrl('http://ruview.example')).toEqual(expect.objectContaining({ valid: false, error: expect.stringContaining('require HTTPS') }));
    expect(normalizeNlosServerUrl('http://fc-public.example').valid).toBe(false);
    expect(normalizeNlosServerUrl('https://ruview.example')).toEqual({ valid: true, normalized: 'https://ruview.example' });
  });

  it('does not weaken origin-only credential and path protection on LAN HTTP', () => {
    for (const unsafe of ['http://user:pass@192.168.1.166:3000', 'http://192.168.1.166:3000/path', 'http://192.168.1.166:3000?token=x']) {
      expect(normalizeNlosServerUrl(unsafe).valid).toBe(false);
    }
  });
});
