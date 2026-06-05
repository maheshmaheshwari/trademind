import axios, { type AxiosError, type AxiosInstance, type InternalAxiosRequestConfig } from 'axios';

const TOKEN_KEY = 'trademind_token';

const onRequest = (config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token && !config.headers['Authorization']) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  if (!(config.data instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json; charset=UTF-8';
  }
  return config;
};

const onResponseError = (error: AxiosError | Error): Promise<AxiosError> => {
  if (axios.isAxiosError(error) && error.response?.status === 401) {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem('trademind_user');
    window.dispatchEvent(new Event('trademind:unauthorized'));
  }
  return Promise.reject(error);
};

const tradeMindInterceptor = (instance: AxiosInstance): AxiosInstance => {
  instance.interceptors.request.use(onRequest, onResponseError);
  instance.interceptors.response.use((res) => res, onResponseError);
  return instance;
};

export default tradeMindInterceptor;
