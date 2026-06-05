import type { BaseQueryFn } from '@reduxjs/toolkit/query';
import axios from 'axios';
import type { AxiosError, AxiosRequestConfig } from 'axios';
import tradeMindInterceptor from './tradeMindInterceptor';

// Single shared axios instance with auth interceptor attached
const axiosInstance = tradeMindInterceptor(axios.create());

const tradeMindBaseQuery = (
  baseURL: string,
): BaseQueryFn<
  {
    url: string;
    method?: AxiosRequestConfig['method'];
    data?: AxiosRequestConfig['data'];
    params?: AxiosRequestConfig['params'];
    headers?: AxiosRequestConfig['headers'];
  },
  unknown,
  unknown
> =>
  async ({ url, method = 'GET', data, params, headers }) => {
    try {
      const response = await axiosInstance({
        url: baseURL + url,
        method,
        data,
        params,
        headers,
      });
      return { data: response.data };
    } catch (axiosError) {
      const err = axiosError as AxiosError;
      return {
        error: {
          status: err?.response?.status,
          data: err?.response?.data || err?.message,
        },
      };
    }
  };

export default tradeMindBaseQuery;
