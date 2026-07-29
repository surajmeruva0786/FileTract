import React from 'react';
import { View, StyleSheet } from 'react-native';
import { colors } from '../theme';

// Cheap approximation of the two radial-gradient glows in the source design
// (FileTract-app-frontend/FileTract.dc.html) using soft, low-opacity circles —
// avoids pulling in react-native-svg just for a background decoration.
export default function GlowBackground() {
  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      <View style={styles.indigoGlow} />
      <View style={styles.violetGlow} />
    </View>
  );
}

const styles = StyleSheet.create({
  indigoGlow: {
    position: 'absolute',
    top: -90,
    right: -90,
    width: 340,
    height: 340,
    borderRadius: 170,
    backgroundColor: colors.glowIndigo,
  },
  violetGlow: {
    position: 'absolute',
    bottom: -110,
    left: -110,
    width: 360,
    height: 360,
    borderRadius: 180,
    backgroundColor: colors.glowViolet,
  },
});
