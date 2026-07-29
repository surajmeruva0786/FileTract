// Design language ported from FileTract-app-frontend/FileTract.dc.html
// (the Claude-designed dark violet UI concept for the mobile app).
import {
  InstrumentSans_600SemiBold,
  InstrumentSans_700Bold,
} from '@expo-google-fonts/instrument-sans';
import {
  DMSans_400Regular,
  DMSans_500Medium,
  DMSans_600SemiBold,
  DMSans_700Bold,
} from '@expo-google-fonts/dm-sans';
import {
  JetBrainsMono_400Regular,
  JetBrainsMono_500Medium,
} from '@expo-google-fonts/jetbrains-mono';

export const colors = {
  background: '#0C0714',
  surface: '#150F22',
  chip: '#1A1428',
  border: 'rgba(255,255,255,0.08)',
  borderStrong: 'rgba(255,255,255,0.16)',

  text: '#FFFFFF',
  textDim: 'rgba(255,255,255,0.55)',
  textFaint: 'rgba(255,255,255,0.4)',
  textGhost: 'rgba(255,255,255,0.32)',

  violetLight: '#C4B5FD',
  violet: '#A78BFA',
  violetDeep: '#8B5CF6',

  glowIndigo: 'rgba(99,102,241,0.30)',
  glowViolet: 'rgba(139,92,246,0.28)',

  success: '#34D399',
  successDeep: '#10B981',
  warning: '#FBBF24',
  danger: '#F87171',
  dangerDeep: '#EF4444',
};

export const gradients = {
  primary: ['#6366F1', '#7C3AED'],
  orb: ['#8B5CF6', '#4C1D95'],
  success: ['#10B981', '#059669'],
};

export const fonts = {
  display: 'InstrumentSans_700Bold',
  displaySemi: 'InstrumentSans_600SemiBold',
  body: 'DMSans_400Regular',
  bodyMedium: 'DMSans_500Medium',
  bodySemi: 'DMSans_600SemiBold',
  bodyBold: 'DMSans_700Bold',
  mono: 'JetBrainsMono_400Regular',
  monoMedium: 'JetBrainsMono_500Medium',
};

export const fontAssets = {
  InstrumentSans_600SemiBold,
  InstrumentSans_700Bold,
  DMSans_400Regular,
  DMSans_500Medium,
  DMSans_600SemiBold,
  DMSans_700Bold,
  JetBrainsMono_400Regular,
  JetBrainsMono_500Medium,
};

// Maps the backend's real quality/confidence signals onto the design's badge language.
export function qualityBadge(quality) {
  switch (quality) {
    case 'reliable':
      return { label: 'Reliable', bg: 'rgba(16,185,129,0.14)', color: colors.success };
    case 'good':
      return { label: 'Good', bg: 'rgba(16,185,129,0.14)', color: colors.success };
    case 'uncertain':
      return { label: 'Review', bg: 'rgba(245,158,11,0.14)', color: colors.warning };
    case 'low-quality':
      return { label: 'Low', bg: 'rgba(239,68,68,0.14)', color: colors.danger };
    default:
      return null;
  }
}
