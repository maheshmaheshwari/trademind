import axios, { type AxiosError, type AxiosInstance, type InternalAxiosRequestConfig } from 'axios';

const TOKEN_KEY = 'trademind_token';

// Audit H11: an unchecked "remember me" stores the token in sessionStorage
// only (see AuthPage.tsx) — this interceptor used to read localStorage
// exclusively, so every request after such a login went out unauthenticated,
// triggering a 401 and an immediate auto-logout. Matches api.ts's existing
// dual-storage lookup.
const getToken = (): string | null =>
  localStorage.getItem(TOKEN_KEY) ?? sessionStorage.getItem(TOKEN_KEY);

const onRequest = (config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
  const token = getToken();
  if (token && !config.headers['Authorization']) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  if (!(config.data instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json; charset=UTF-8';
  }
  return config;
};

// Audit M21: concurrent in-flight requests each independently dispatched
// their own unauthorized/logout event on a 401 — idempotent but redundant,
// and with no guard a burst of failing requests could fire many duplicate
// logout/redirect cycles. This single-flight guard ensures only the first
// 401 in a burst triggers the logout dispatch; it resets once the token is
// cleared (i.e. on the next page load / re-login).
let loggingOut = false;

export function resetLoggingOut(): void {
  loggingOut = false;
}

const onResponseError = (error: AxiosError | Error): Promise<AxiosError> => {
  if (axios.isAxiosError(error) && error.response?.status === 401) {
    if (!loggingOut) {
      loggingOut = true;
      localStorage.removeItem(TOKEN_KEY);
      sessionStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem('trademind_user');
      sessionStorage.removeItem('trademind_user');
      window.dispatchEvent(new Event('trademind:unauthorized'));
    }
  }
  return Promise.reject(error);
};

const tradeMindInterceptor = (instance: AxiosInstance): AxiosInstance => {
  instance.interceptors.request.use(onRequest, onResponseError);
  instance.interceptors.response.use((res) => res, onResponseError);
  return instance;
};

export default tradeMindInterceptor;
