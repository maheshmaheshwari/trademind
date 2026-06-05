import dayjs from 'dayjs';
import { Tooltip } from '@mui/material';

const DEFAULT_FORMAT = 'DD MMM YYYY';

function isVoid(val: unknown): val is null | undefined | '' {
  return val === null || val === undefined || val === '';
}

function formatDate(date: string, format: string, inputFormat?: string, showDetailed?: boolean): string {
  try {
    if (isVoid(date)) return '-';
    const parsed = inputFormat ? dayjs(date, inputFormat) : dayjs(date);
    if (!parsed.isValid()) return date || '-';
    const hasTime = parsed.hour() !== 0 || parsed.minute() !== 0 || parsed.second() !== 0;
    if (showDetailed && hasTime) return parsed.format('DD MMM YYYY [at] h:mm A');
    return parsed.format(format);
  } catch {
    return date || '-';
  }
}

function detailedDate(date: string, inputFormat?: string): string | null {
  try {
    if (isVoid(date)) return null;
    const parsed = inputFormat ? dayjs(date, inputFormat) : dayjs(date);
    if (!parsed.isValid()) return null;
    const hasTime = /\d{2}:\d{2}/.test(date);
    return hasTime
      ? parsed.format('dddd, MMMM D, YYYY [at] h:mm A')
      : parsed.format('dddd, MMMM D, YYYY');
  } catch {
    return null;
  }
}

type DateComponentProps = {
  id?: string;
  inputDate: string;
  format?: string;
  inputFormat?: string;
  showDetailed?: boolean;
  showTooltip?: boolean;
};

export function DateComponent({
  id,
  inputDate,
  format = DEFAULT_FORMAT,
  inputFormat,
  showDetailed = false,
  showTooltip = true,
}: DateComponentProps) {
  const formatted = formatDate(inputDate, format, inputFormat, showDetailed);
  const tooltip = detailedDate(inputDate, inputFormat);

  if (showTooltip && tooltip) {
    return (
      <Tooltip title={tooltip} placement="top">
        <span id={id}>{formatted}</span>
      </Tooltip>
    );
  }

  return <span id={id}>{formatted}</span>;
}
