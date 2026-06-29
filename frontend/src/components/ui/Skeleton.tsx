interface SkeletonProps {
  w?: string | number;
  h?: number;
  rounded?: string;
  className?: string;
}

export function Skeleton({ w = '100%', h = 14, rounded = '7px', className = '' }: SkeletonProps) {
  return (
    <span
      className={`skel block ${className}`}
      style={{ width: w, height: h, borderRadius: rounded, display: 'block' }}
    />
  );
}

// Widths cycled across numeric cells so they don't look uniform
const NUM_WIDTHS = ['52%', '62%', '45%', '58%', '40%', '55%', '48%', '60%'];

export function SkeletonRows({ cols, rows = 6 }: { cols: number; rows?: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, i) => (
        <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
          {Array.from({ length: cols }).map((_, j) => {
            const isFirst  = j === 0;
            const isLast   = j === cols - 1;

            if (isFirst) {
              // Symbol cell: circle avatar + name line + sector line
              return (
                <td key={j} style={{ padding: '10px 14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span className="skel" style={{ width: 34, height: 34, borderRadius: 10, flexShrink: 0 }} />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5, flex: 1, minWidth: 0 }}>
                      <Skeleton w={`${48 + ((i * 13) % 28)}%`} h={12} rounded="5px" />
                      <Skeleton w={`${32 + ((i * 7) % 22)}%`} h={10} rounded="4px" />
                    </div>
                  </div>
                </td>
              );
            }

            if (isLast) {
              // Actions cell: button-shaped skeleton
              return (
                <td key={j} style={{ padding: '10px 14px', textAlign: 'right' }}>
                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6 }}>
                    <span className="skel" style={{ width: 70, height: 30, borderRadius: 9 }} />
                    <span className="skel" style={{ width: 58, height: 30, borderRadius: 9 }} />
                  </div>
                </td>
              );
            }

            // Numeric cells: narrow right-aligned bars of varying widths
            const w = NUM_WIDTHS[(i + j) % NUM_WIDTHS.length];
            return (
              <td key={j} style={{ padding: '10px 14px', textAlign: 'right' }}>
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <Skeleton w={w} h={12} rounded="5px" />
                </div>
              </td>
            );
          })}
        </tr>
      ))}
    </>
  );
}
