import type { ReactNode } from 'react';

interface CardProps {
  title?: string;
  sub?: string;
  icon?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  pad?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

export function Card({ title, sub, icon, right, children, pad = true, className = '', style }: CardProps) {
  return (
    <div className={`bg-surface border border-line rounded-card relative ${className}`} style={style}>
      {(title || right) && (
        <div className="flex items-center justify-between gap-3 dp-head border-b border-line">
          <div className="flex flex-col gap-0">
            <h3 className="text-[14.5px] font-semibold flex items-center gap-2 text-ink m-0 whitespace-nowrap">
              {icon && <span className="text-ink-3">{icon}</span>}
              {title}
            </h3>
            {sub && <span className="text-[12px] text-ink-3 mt-0.5">{sub}</span>}
          </div>
          {right}
        </div>
      )}
      {pad ? <div className="dp">{children}</div> : children}
    </div>
  );
}
