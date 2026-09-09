import type { ReactNode } from 'react';
import { cn } from '../../lib/earlyWarningsUtils';

export type Tone = 'critical' | 'high' | 'medium' | 'low' | 'info' | 'success';

const tones: Record<Tone, string> = {
  critical: 'bg-critical-soft text-critical', high: 'bg-warn-soft text-warn',
  medium: 'bg-warn-soft/70 text-warn', low: 'bg-muted text-muted-foreground',
  info: 'bg-info-soft text-info', success: 'bg-success-soft text-success',
};

export function Pill({ tone, children, className }: { tone: Tone; children: ReactNode; className?: string }) {
  return <span className={cn('inline-flex items-center whitespace-nowrap rounded-md px-2 py-0.5 text-xs font-semibold', tones[tone], className)}>{children}</span>;
}

export function RiskText({ level, score }: { level: string; score: number }) {
  const color = level === 'Critical' ? 'text-critical' : level === 'High' ? 'text-warn' : 'text-warn/80';
  return <div className={cn('text-sm font-semibold leading-tight', color)}>{level}<div className="text-xs font-medium">({score})</div></div>;
}
