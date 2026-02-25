import { createApi } from '@reduxjs/toolkit/query/react';
import tradeMindBaseQuery from './tradeMindBaseQuery';

export const tradeMindApiClient = (reducerPath: string, baseURL?: string) =>
  createApi({
    baseQuery: tradeMindBaseQuery(baseURL as string),
    endpoints: () => ({}),
    reducerPath: reducerPath,
    tagTypes: ['Signals', 'Portfolio', 'Sectors'],
  });
