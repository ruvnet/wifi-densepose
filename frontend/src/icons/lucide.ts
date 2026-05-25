/**
 * Minimal Lucide icon wrapper.
 * Import only the icons used by HOMECORE components — Vite tree-shakes the rest.
 */

export {
  Activity,
  BarChart3,
  Book,
  ChevronRight,
  Grid2X2,
  Home,
  LayoutDashboard,
  Settings,
  Shield,
  Sun,
  Wifi,
  Zap,
} from 'lucide';

/** Re-export the icon node type for consumers that need it. */
export type { IconNode as LucideIconNode } from 'lucide';

/**
 * Render a Lucide icon as an SVG string suitable for Lit's `unsafeHTML`.
 * Each icon is 24×24, no fill, stroke = currentColor, stroke-width = 2.
 */
export function iconSvg(
  paths: string,
  { size = 24, label }: { size?: number; label?: string } = {},
): string {
  const ariaAttrs = label
    ? `role="img" aria-label="${label}"`
    : `aria-hidden="true"`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}"
    viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
    ${ariaAttrs}>${paths}</svg>`;
}
