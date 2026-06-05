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

export function SkeletonRows({ cols, rows = 6 }: { cols: number; rows?: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, i) => (
        <tr key={i}>
          {Array.from({ length: cols }).map((_, j) => (
            <td key={j} className="px-3.5 py-3">
              <Skeleton w={j === 0 ? '70%' : '55%'} h={j === 0 ? 28 : 13} />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}
