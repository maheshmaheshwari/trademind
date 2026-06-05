import { ReactNode } from 'react';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import type { DatePickerProps } from '@mui/x-date-pickers/DatePicker';
import type { Dayjs } from 'dayjs';

const DEFAULT_FORMAT = 'DD MMM YYYY';

interface DatePickerInputProps extends Omit<DatePickerProps<Dayjs>, 'label'> {
  label?: ReactNode;
  showLabel?: boolean;
  errorMessage?: string;
  isError?: boolean;
  requiredLabel?: boolean;
  wrapperClassName?: string;
  labelClassName?: string;
}

export function DatePickerInput({
  label,
  showLabel = true,
  errorMessage,
  isError = false,
  requiredLabel = false,
  wrapperClassName,
  labelClassName,
  disabled = false,
  format = DEFAULT_FORMAT,
  ...rest
}: DatePickerInputProps) {
  return (
    <div className={wrapperClassName} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {showLabel && label && (
        <div className={labelClassName}>
          <label style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-2)' }}>
            {label}
            {requiredLabel && (
              <span style={{ color: 'var(--red)', fontSize: 14, marginLeft: 3 }}>*</span>
            )}
          </label>
        </div>
      )}

      <DatePicker<Dayjs>
        format={format}
        disabled={disabled}
        {...rest}
        slotProps={{
          textField: {
            size: 'small',
            error: isError,
            disabled,
            sx: {
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
                '&.Mui-disabled': {
                  opacity: 0.55,
                },
              },
              '& .MuiInputBase-input': {
                color: 'var(--text)',
                '&::placeholder': { color: 'var(--text-3)', opacity: 1 },
              },
              '& .MuiSvgIcon-root': {
                color: 'var(--text-3)',
              },
            },
          },
          popper: {
            sx: {
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
                '&.Mui-selected': {
                  background: 'var(--accent) !important',
                  color: '#fff',
                },
                '&.MuiPickersDay-today': {
                  border: '1px solid var(--accent)',
                },
              },
              '& .MuiDayCalendar-weekDayLabel': { color: 'var(--text-3)' },
              '& .MuiPickersCalendarHeader-label': { color: 'var(--text)', fontWeight: 600 },
              '& .MuiPickersArrowSwitcher-button': { color: 'var(--text-2)' },
              '& .MuiPickersYear-yearButton': {
                color: 'var(--text)',
                '&.Mui-selected': { background: 'var(--accent)', color: '#fff' },
              },
              '& .MuiPickersMonth-monthButton': {
                color: 'var(--text)',
                '&.Mui-selected': { background: 'var(--accent)', color: '#fff' },
              },
            },
          },
          ...rest.slotProps,
        }}
      />

      {isError && errorMessage && (
        <span style={{ fontSize: 12, color: 'var(--red)', marginTop: 2 }}>
          {errorMessage}
        </span>
      )}
    </div>
  );
}
