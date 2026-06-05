import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import type { Dayjs } from 'dayjs';

const DEFAULT_FORMAT = 'DD MMM YYYY';

const inputSx = (isError?: boolean) => ({
  width: '100%',
  '& .MuiOutlinedInput-root': {
    height: 44,
    borderRadius: '11px',
    background: 'var(--surface-2)',
    color: 'var(--text)',
    fontFamily: 'inherit',
    fontSize: 14,
    '& fieldset': {
      borderColor: isError ? 'var(--red)' : 'var(--border-strong)',
    },
    '&:hover fieldset': {
      borderColor: isError ? 'var(--red)' : 'var(--accent)',
    },
    '&.Mui-focused fieldset': {
      borderColor: isError ? 'var(--red)' : 'var(--accent)',
    },
    '&.Mui-disabled': { opacity: 0.55 },
  },
  '& .MuiInputBase-input': {
    color: 'var(--text)',
    '&::placeholder': { color: 'var(--text-3)', opacity: 1 },
  },
  '& .MuiSvgIcon-root': { color: 'var(--text-3)' },
});

const popperSx = {
  '& .MuiPaper-root': {
    background: 'var(--surface)',
    border: '1px solid var(--border-strong)',
    borderRadius: '14px',
    boxShadow: 'var(--shadow-lg)',
    color: 'var(--text)',
  },
  '& .MuiPickersDay-root': {
    color: 'var(--text)',
    borderRadius: '8px',
    '&:hover': { background: 'var(--surface-hover)' },
    '&.Mui-selected': { background: 'var(--accent) !important', color: '#fff' },
    '&.MuiPickersDay-today': { border: '1px solid var(--accent)' },
  },
  '& .MuiDayCalendar-weekDayLabel': { color: 'var(--text-3)' },
  '& .MuiPickersCalendarHeader-label': { color: 'var(--text)', fontWeight: 600 },
  '& .MuiPickersArrowSwitcher-button': { color: 'var(--text-2)' },
  '& .MuiPickersYear-yearButton': {
    color: 'var(--text)',
    '&.Mui-selected': { background: 'var(--accent)', color: '#fff' },
  },
};

interface DateRangeInputProps {
  label?: string;
  showLabel?: boolean;
  errorMessage?: string;
  isError?: boolean;
  isMandatory?: boolean;
  wrapperClassName?: string;
  format?: string;
  value?: [Dayjs | null, Dayjs | null];
  onChange?: (value: [Dayjs | null, Dayjs | null]) => void;
  startLabel?: string;
  endLabel?: string;
  disabled?: boolean;
  disableFuture?: boolean;
  disablePast?: boolean;
  showYearRange?: number;
}

export function DateRangeInput({
  label,
  showLabel = true,
  errorMessage,
  isError = false,
  isMandatory = false,
  wrapperClassName,
  format = DEFAULT_FORMAT,
  value = [null, null],
  onChange,
  startLabel = 'From',
  endLabel = 'To',
  disabled = false,
  disableFuture,
  disablePast,
  showYearRange,
}: DateRangeInputProps) {
  const [start, end] = value;

  function shouldDisable(date: Dayjs) {
    if (showYearRange !== undefined) return date.year() !== showYearRange;
    return false;
  }

  return (
    <div className={wrapperClassName} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {showLabel && label && (
        <label style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-2)' }}>
          {label}
          {isMandatory && (
            <span style={{ color: 'var(--red)', fontSize: 14, marginLeft: 3 }}>*</span>
          )}
        </label>
      )}

      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={{ fontSize: 11.5, color: 'var(--text-3)' }}>{startLabel}</span>
          <DatePicker<Dayjs>
            format={format}
            value={start}
            onChange={d => onChange?.([d, end])}
            disabled={disabled}
            disableFuture={disableFuture}
            disablePast={disablePast}
            maxDate={end ?? undefined}
            shouldDisableDate={showYearRange !== undefined ? shouldDisable : undefined}
            slotProps={{
              textField: { size: 'small', error: isError, sx: inputSx(isError) },
              popper: { sx: popperSx },
            }}
          />
        </div>

        <span style={{ color: 'var(--text-3)', fontSize: 16, paddingTop: 18, flexShrink: 0 }}>—</span>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={{ fontSize: 11.5, color: 'var(--text-3)' }}>{endLabel}</span>
          <DatePicker<Dayjs>
            format={format}
            value={end}
            onChange={d => onChange?.([start, d])}
            disabled={disabled}
            disableFuture={disableFuture}
            disablePast={disablePast}
            minDate={start ?? undefined}
            shouldDisableDate={showYearRange !== undefined ? shouldDisable : undefined}
            slotProps={{
              textField: { size: 'small', error: isError, sx: inputSx(isError) },
              popper: { sx: popperSx },
            }}
          />
        </div>
      </div>

      {isError && errorMessage && (
        <span style={{ fontSize: 12, color: 'var(--red)', marginTop: 2 }}>
          {errorMessage}
        </span>
      )}
    </div>
  );
}
