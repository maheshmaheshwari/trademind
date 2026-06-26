import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

// Audit Low item: no ErrorBoundary anywhere in the frontend meant any
// render-time exception (e.g. from the heavy use of `as any` casts on API
// responses) blanked the entire app with no recovery UI.
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Unhandled render error:', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen bg-[var(--bg)] flex items-center justify-center text-[var(--text)] p-6">
          <div className="max-w-md text-center flex flex-col gap-3">
            <h1 className="text-xl font-bold">Something went wrong</h1>
            <p className="text-[var(--text-2)] text-sm">
              The page hit an unexpected error. Try reloading — if it keeps happening, please report it.
            </p>
            <button
              onClick={() => { this.setState({ error: null }); window.location.reload(); }}
              className="mt-2 self-center px-4 py-2 rounded-lg bg-[var(--accent)] text-white text-sm font-semibold"
            >
              Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
