import { motion } from 'motion/react';
import { CALIBRATION, TARGETS } from '../../data/results';
import { easeOut } from './util';

/*
  Reliability plot. For each soil property we plot empirical interval coverage
  against the nominal level it targets (68, 90-conformal, 95). Points on the
  dashed diagonal are perfectly calibrated; the heteroscedastic head slightly
  under-covers at 68% and the split-conformal step pulls 90% back onto the line.
*/

const PAD_L = 46;
const PAD_B = 38;
const SIZE = 360;
const plot = SIZE - PAD_L - 14;

const sx = (v: number) => PAD_L + ((v - 0.5) / 0.5) * plot;
const sy = (v: number) => SIZE - PAD_B - ((v - 0.5) / 0.5) * plot;

const POINTS = CALIBRATION.flatMap((c) => [
  { t: c.key, nominal: 0.68, emp: c.cov68, kind: 'gaussian' as const },
  { t: c.key, nominal: 0.9, emp: c.conformal90, kind: 'conformal' as const },
  { t: c.key, nominal: 0.95, emp: c.cov95, kind: 'gaussian' as const },
]);

const COLOR: Record<string, string> = {
  soc: 'var(--color-moss-600)',
  ph: 'var(--color-moss-400)',
  sand: 'var(--color-moss-300)',
  clay: 'var(--color-clay)',
  bd: 'var(--color-clay-soft)',
};

export default function CalibrationChart() {
  const grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  return (
    <div className="not-prose w-full">
      <div className="rounded-2xl border border-line bg-card p-3 sm:p-5">
        <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-start">
          <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full max-w-[360px]" role="img"
            aria-label="Calibration: empirical vs nominal coverage">
            {grid.map((g) => (
              <g key={g}>
                <line x1={sx(g)} x2={sx(g)} y1={sy(0.5)} y2={sy(1)}
                  stroke="var(--color-line)" strokeWidth={1} strokeDasharray="2 4" />
                <line x1={sx(0.5)} x2={sx(1)} y1={sy(g)} y2={sy(g)}
                  stroke="var(--color-line)" strokeWidth={1} strokeDasharray="2 4" />
                <text x={sx(g)} y={SIZE - PAD_B + 16} textAnchor="middle"
                  className="fill-ink-faint" style={{ fontFamily: 'var(--font-mono)', fontSize: 9.5 }}>
                  {g.toFixed(1)}
                </text>
                <text x={PAD_L - 8} y={sy(g) + 3} textAnchor="end"
                  className="fill-ink-faint" style={{ fontFamily: 'var(--font-mono)', fontSize: 9.5 }}>
                  {g.toFixed(1)}
                </text>
              </g>
            ))}
            {/* ideal diagonal */}
            <line x1={sx(0.5)} y1={sy(0.5)} x2={sx(1)} y2={sy(1)}
              stroke="var(--color-ink-faint)" strokeWidth={1.4} strokeDasharray="5 4" />
            <text x={sx(0.86)} y={sy(0.9) - 8} className="fill-ink-faint"
              style={{ fontFamily: 'var(--font-mono)', fontSize: 9.5 }} transform={`rotate(-45 ${sx(0.86)} ${sy(0.9)})`}>
              ideal
            </text>

            {POINTS.map((p, i) => (
              <motion.circle
                key={`${p.t}-${p.nominal}`}
                cx={sx(p.nominal)} cy={sy(p.emp)}
                r={p.kind === 'conformal' ? 5 : 4}
                fill={p.kind === 'conformal' ? 'none' : COLOR[p.t]}
                stroke={COLOR[p.t]} strokeWidth={p.kind === 'conformal' ? 2 : 0}
                initial={{ opacity: 0, scale: 0.4 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.02, ease: easeOut }}
              />
            ))}

            <text x={PAD_L + plot / 2} y={SIZE - 4} textAnchor="middle" className="fill-ink-3"
              style={{ fontFamily: 'var(--font-sans)', fontSize: 11 }}>
              nominal coverage
            </text>
            <text x={14} y={sy(0.75)} textAnchor="middle" className="fill-ink-3"
              style={{ fontFamily: 'var(--font-sans)', fontSize: 11 }}
              transform={`rotate(-90 14 ${sy(0.75)})`}>
              empirical coverage
            </text>
          </svg>

          <div className="w-full sm:w-auto">
            <div className="grid grid-cols-2 gap-x-5 gap-y-1.5 sm:grid-cols-1">
              {TARGETS.map((t) => (
                <div key={t.key} className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: COLOR[t.key] }} />
                  <span className="text-sm text-ink-2">{t.label}</span>
                  <span className="font-mono text-[0.7rem] text-ink-faint">{t.long}</span>
                </div>
              ))}
            </div>
            <div className="mt-4 space-y-2 border-t border-line pt-3 font-mono text-[0.7rem] text-ink-3">
              <p className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full bg-moss-400" /> filled · Gaussian σ interval
              </p>
              <p className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full border-2 border-moss-400" /> ring · split-conformal 90%
              </p>
            </div>
          </div>
        </div>
      </div>
      <p className="mt-3 font-mono text-[0.68rem] text-ink-faint">
        source: verification_jepa.json → calibration · coverage averaged over spatial folds
      </p>
    </div>
  );
}
