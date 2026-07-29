import React, { useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  SafeAreaView,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import GlowBackground from '../components/GlowBackground';
import { colors, gradients, fonts } from '../theme';

export default function HomeScreen({ navigation }) {
  const pickFromCamera = useCallback(async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      alert('Camera permission is required to photograph ID cards.');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.95,
      allowsEditing: true,
      aspect: [4, 3],
    });
    if (!result.canceled && result.assets?.[0]) {
      navigation.navigate('Fields', { image: result.assets[0] });
    }
  }, [navigation]);

  const pickFromGallery = useCallback(async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      alert('Photo library permission is required to select images.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.95,
      allowsEditing: true,
      aspect: [4, 3],
    });
    if (!result.canceled && result.assets?.[0]) {
      navigation.navigate('Fields', { image: result.assets[0] });
    }
  }, [navigation]);

  return (
    <View style={styles.root}>
      <GlowBackground />
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor={colors.background} />

        <View style={styles.header}>
          <View style={styles.logoRow}>
            <Ionicons name="aperture-outline" size={26} color={colors.violet} />
            <Text style={styles.logo}>FileTract</Text>
          </View>
          <Text style={styles.tagline}>Document &amp; ID Intelligence</Text>

          <View style={styles.badge}>
            <Ionicons name="sparkles" size={12} color={colors.violetLight} />
            <Text style={styles.badgeText}>AI-POWERED</Text>
          </View>
        </View>

        <View style={styles.hero}>
          <View style={styles.iconGlow}>
            <LinearGradient colors={gradients.orb} style={styles.iconOrb}>
              <Ionicons name="scan" size={44} color="#fff" />
            </LinearGradient>
          </View>

          <View style={styles.heroText}>
            <Text style={styles.heroTitle}>Precision Extraction,</Text>
            <Text style={[styles.heroTitle, styles.heroTitleAccent]}>Engineered for Trust.</Text>
          </View>
          <Text style={styles.heroSubtitle}>
            Capture any document or identity card and FileTract extracts structured, verified
            fields in seconds — built for secure, compliant workflows.
          </Text>
        </View>

        <View style={styles.actions}>
          <TouchableOpacity activeOpacity={0.85} onPress={pickFromCamera}>
            <LinearGradient colors={gradients.primary} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.btnPrimary}>
              <Ionicons name="camera" size={19} color="#fff" />
              <Text style={styles.btnPrimaryText}>Take Photo</Text>
            </LinearGradient>
          </TouchableOpacity>

          <TouchableOpacity style={styles.btnSecondary} activeOpacity={0.8} onPress={pickFromGallery}>
            <Ionicons name="images" size={19} color="rgba(255,255,255,0.85)" />
            <Text style={styles.btnSecondaryText}>Upload from Gallery</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.settingsLink} onPress={() => navigation.navigate('Settings')}>
          <Text style={styles.settingsText}>Settings</Text>
        </TouchableOpacity>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
  },
  container: {
    flex: 1,
    paddingHorizontal: 26,
    paddingTop: 24,
    paddingBottom: 20,
  },
  header: {
    alignItems: 'center',
  },
  logoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  logo: {
    fontFamily: fonts.display,
    fontSize: 22,
    color: colors.text,
    letterSpacing: -0.3,
  },
  tagline: {
    fontFamily: fonts.body,
    marginTop: 6,
    fontSize: 12.5,
    color: colors.textDim,
    letterSpacing: 0.4,
  },
  badge: {
    marginTop: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 999,
    backgroundColor: 'rgba(167,139,250,0.12)',
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.28)',
  },
  badgeText: {
    fontFamily: fonts.bodySemi,
    fontSize: 11.5,
    letterSpacing: 1,
    color: colors.violetLight,
  },
  hero: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 22,
    paddingHorizontal: 4,
    paddingVertical: 20,
  },
  iconGlow: {
    width: 116,
    height: 116,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconOrb: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
  },
  heroText: {
    alignItems: 'center',
  },
  heroTitle: {
    fontFamily: fonts.display,
    fontSize: 27,
    lineHeight: 32,
    letterSpacing: -0.3,
    color: colors.text,
    textAlign: 'center',
  },
  heroTitleAccent: {
    color: colors.violet,
  },
  heroSubtitle: {
    fontFamily: fonts.body,
    fontSize: 14.5,
    lineHeight: 23,
    color: colors.textDim,
    textAlign: 'center',
    maxWidth: 320,
  },
  actions: {
    gap: 12,
    marginTop: 8,
  },
  btnPrimary: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 16,
    borderRadius: 16,
  },
  btnPrimaryText: {
    fontFamily: fonts.bodySemi,
    color: '#fff',
    fontSize: 15.5,
  },
  btnSecondary: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 15,
    borderRadius: 16,
    borderWidth: 1.5,
    borderColor: colors.borderStrong,
  },
  btnSecondaryText: {
    fontFamily: fonts.bodySemi,
    color: 'rgba(255,255,255,0.85)',
    fontSize: 15.5,
  },
  settingsLink: {
    alignSelf: 'center',
    marginTop: 20,
    padding: 6,
  },
  settingsText: {
    fontFamily: fonts.bodyMedium,
    color: colors.textFaint,
    fontSize: 13.5,
  },
});
