import { useState } from 'react';
import { cn } from '../../lib/earlyWarningsUtils';

export function EarlyWarningsSwitch({ defaultChecked = false, checked: controlledChecked, onCheckedChange }: { defaultChecked?: boolean; checked?: boolean; onCheckedChange?: (checked: boolean) => void }) {
  const [checked, setChecked] = useState(defaultChecked);
  const value = controlledChecked ?? checked;
  const update = () => { const next = !value; if (controlledChecked === undefined) setChecked(next); onCheckedChange?.(next); };
  return <button type="button" role="switch" aria-checked={value} onClick={update} className={cn('peer inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background', value ? 'bg-primary' : 'bg-input')}><span className={cn('pointer-events-none block h-4 w-4 rounded-full bg-background shadow-lg ring-0 transition-transform', value && 'translate-x-4')} /></button>;
}
