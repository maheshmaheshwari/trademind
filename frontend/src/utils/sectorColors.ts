/**
 * A stable colour per sector, for anything that draws sectors side by side.
 *
 * Sector names come from `nifty_constituents`, which uses the exchange's own
 * labels ("Fast Moving Consumer Goods", "Oil Gas & Consumable Fuels", …), and
 * that list grows as the index is reconstituted. A hardcoded name → colour map
 * therefore cannot be complete, and an unmapped sector rendering as `undefined`
 * makes a donut slice invisible — the chart then shows a total that its visible
 * slices do not add up to.
 *
 * So the colour is derived from the name instead: every sector gets one, the
 * same one on every screen and every reload, without a list to maintain.
 */

// Distinct at donut-slice size in both themes. Deliberately no near-neighbours
// (two blues, two greens) — adjacent slices have to be told apart.
const PALETTE = [
  '#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EC4899', '#14B8A6',
  '#F97316', '#6366F1', '#EAB308', '#0EA5E9', '#A78BFA', '#EF4444',
];

/** FNV-1a — small, dependency-free, and well spread over short strings. */
function hash(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export function sectorColor(sector: string | null | undefined): string {
  if (!sector) return '#94A3B8';   // Unclassified — grey, never a data colour
  return PALETTE[hash(sector) % PALETTE.length];
}

/**
 * Colours for a whole set of sectors shown together.
 *
 * Hashing alone can collide, and two identically-coloured slices in one donut
 * is the failure this is meant to prevent — so a collision inside a single
 * chart is nudged to the next free colour. Order matters (largest slice first
 * keeps its hashed colour), which is why this takes the list rather than being
 * called per item.
 */
export function sectorColors(sectors: (string | null | undefined)[]): string[] {
  const used = new Set<string>();
  return (sectors ?? []).map(s => {
    let c = sectorColor(s);
    if (used.has(c)) {
      const start = PALETTE.indexOf(c);
      for (let i = 1; i <= PALETTE.length; i++) {
        const next = PALETTE[(start + i) % PALETTE.length];
        if (!used.has(next)) { c = next; break; }
      }
    }
    used.add(c);
    return c;
  });
}
