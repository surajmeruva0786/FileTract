import React, { useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  SafeAreaView,
  Image,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';

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
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={colors.background} />

      <View style={styles.header}>
        <Text style={styles.logo}>
          File<Text style={styles.logoAccent}>Tract</Text>
        </Text>
        <Text style={styles.tagline}>Patent-Pending ID Extraction</Text>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>AI-POWERED • PATENT PENDING</Text>
        </View>
      </View>

      <View style={styles.hero}>
        <View style={styles.iconContainer}>
          <Ionicons name="id-card" size={80} color={colors.accent} />
        </View>
        <Text style={styles.heroTitle}>Extract Any ID Field</Text>
        <Text style={styles.heroSubtitle}>
          Photograph or upload your Aadhaar card, ID card, or any document.
          FileTract's AI extracts the exact fields you need — directly into Google Sheets.
        </Text>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity style={[styles.btn, styles.btnPrimary]} onPress={pickFromCamera}>
          <Ionicons name="camera" size={24} color={colors.background} />
          <Text style={styles.btnPrimaryText}>Take Photo</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.btn, styles.btnSecondary]} onPress={pickFromGallery}>
          <Ionicons name="images" size={24} color={colors.accent} />
          <Text style={styles.btnSecondaryText}>Upload from Gallery</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={styles.settingsLink} onPress={() => navigation.navigate('Settings')}>
        <Ionicons name="settings-outline" size={18} color={colors.textDim} />
        <Text style={styles.settingsText}>Settings</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const colors = {
  background: '#0a0a12',
  surface: '#12121e',
  accent: '#00e5ff',
  accentGlow: '#00e5ff33',
  text: '#e8e8f0',
  textDim: '#666680',
  border: '#1e1e2e',
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    paddingHorizontal: 24,
  },
  header: {
    alignItems: 'center',
    paddingTop: 40,
    paddingBottom: 20,
  },
  logo: {
    fontSize: 38,
    fontWeight: '900',
    color: '#ffffff',
    letterSpacing: -1,
  },
  logoAccent: {
    color: colors.accent,
  },
  tagline: {
    fontSize: 12,
    color: colors.textDim,
    letterSpacing: 2,
    marginTop: 4,
    textTransform: 'uppercase',
  },
  badge: {
    marginTop: 12,
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: colors.accent + '66',
    backgroundColor: colors.accentGlow,
  },
  badgeText: {
    color: colors.accent,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1.5,
  },
  hero: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 40,
  },
  iconContainer: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: colors.accentGlow,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 32,
    borderWidth: 1,
    borderColor: colors.accent + '44',
  },
  heroTitle: {
    fontSize: 28,
    fontWeight: '800',
    color: colors.text,
    textAlign: 'center',
    marginBottom: 16,
  },
  heroSubtitle: {
    fontSize: 15,
    color: colors.textDim,
    textAlign: 'center',
    lineHeight: 24,
    maxWidth: 300,
  },
  actions: {
    gap: 14,
    marginBottom: 24,
  },
  btn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    paddingVertical: 18,
    borderRadius: 14,
  },
  btnPrimary: {
    backgroundColor: colors.accent,
  },
  btnPrimaryText: {
    color: colors.background,
    fontSize: 17,
    fontWeight: '800',
  },
  btnSecondary: {
    backgroundColor: 'transparent',
    borderWidth: 1.5,
    borderColor: colors.accent,
  },
  btnSecondaryText: {
    color: colors.accent,
    fontSize: 17,
    fontWeight: '700',
  },
  settingsLink: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingBottom: 16,
  },
  settingsText: {
    color: colors.textDim,
    fontSize: 13,
  },
});
