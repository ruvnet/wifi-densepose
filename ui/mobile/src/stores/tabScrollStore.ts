import { useEffect, useRef } from 'react';
import type { ScrollView } from 'react-native';
import { create } from 'zustand';
import type { MainTabsParamList } from '@/navigation/types';

type TabName = keyof MainTabsParamList;

const initialTokens: Record<TabName, number> = {
  Welcome: 0,
  Live: 0,
  Calibration: 0,
  Vitals: 0,
  Zones: 0,
  MAT: 0,
  Settings: 0,
};

type TabScrollState = {
  tokens: Record<TabName, number>;
  requestTop: (tab: TabName) => void;
};

export const useTabScrollStore = create<TabScrollState>((set) => ({
  tokens: initialTokens,
  requestTop: (tab) => set((state) => ({
    tokens: { ...state.tokens, [tab]: state.tokens[tab] + 1 },
  })),
}));

/** Reset a tab's primary vertical scroll surface after every accepted tab tap. */
export const useTabScrollToTop = (tab: TabName) => {
  const ref = useRef<ScrollView>(null);
  const token = useTabScrollStore((state) => state.tokens[tab]);

  useEffect(() => {
    if (token > 0) ref.current?.scrollTo({ y: 0, animated: false });
  }, [token]);

  return ref;
};
