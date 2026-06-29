import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  SafeAreaView,
  StatusBar,
  TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { processImage } from '../services/api';
import { getBackendUrl as getStoredUrl } from '../services/storage';
import { setBackendUrl } from '../services/api';

const STAGES = [
  { id: 1, label: 'Uploading Image', icon: 'cloud-upload' },
  { id: 2, label: 'Baseline OCR', icon: 'scan' },
  { id: 3, label: 'Confidence Analysis', icon: 'analytics' },
  { id: 4, label: 'Adaptive Re-OCR', icon: 'refresh-circle' },
  { id: 5, label: 'AI Field Extraction', icon: 'sparkles' },
];

export default function ProcessingScreen({ navigation, route }) {
  const { image, fields, pipeline } = route.params;
  const [currentStage, setCurrentStage] = useState(0);
  const [error, setError] = useState(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const stageCount = pipeline === 'patent' ? 5 : 2;

  useEffect(() => {
    // Pulse animation
    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.15, duration: 800, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
      ])
    );
    pulse.start();
    return () => pulse.stop();
  }, []);

  useEffect(() => {
    run();
  }, []);

  const run = async () => {
    try {
      // Apply stored backend URL if set
      const storedUrl = await getStoredUrl();
      if (storedUrl) setBackendUrl(storedUrl);

      setCurrentStage(1);
      const results = await processImage(image.uri, fields, pipeline, (stage) => {
        setCurrentStage(stage);
      });
      navigation.replace('Preview', { image, fields, results, pipeline });
    } catch (err) {
      setError(err.message || 'Processing failed. Check your backend URL in Settings.');
    }
  };

  const stageList = pipeline === 'patent' ? STAGES : STAGES.slice(0, 2);

  if (error) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor={colors.background} />
        <View style={styles.errorBox}>
          <Ionicons name="warning" size={48} color="#ff4444" />
          <Text style={styles.errorTitle}>Processing Failed</Text>
          <Text style={styles.errorMsg}>{error}</Text>
          <TouchableOpacity style={styles.retryBtn} onPress={() => { setError(null); run(); }}>
            <Text style={styles.retryBtnText}>Retry</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.backLink} onPress={() => navigation.goBack()}>
            <Text style={styles.backLinkText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={colors.background} />

      <View style={styles.center}>
        <Animated.View style={[styles.orb, { transform: [{ scale: pulseAnim }] }]}>
          <Ionicons name="scan" size={48} color={colors.accent} />
        </Animated.View>

        <Text style={styles.processingTitle}>Extracting Fields</Text>
        <Text style={styles.processingSubtitle}>
          {pipeline === 'patent' ? 'Patent Pipeline — 5 stages' : 'Standard Pipeline'}
        </Text>

        <View style={styles.stageList}>
          {stageList.map((stage) => {
            const done = currentStage > stage.id;
            const active = currentStage === stage.id;
            return (
              <View key={stage.id} style={styles.stageRow}>
                <View
                  style={[
                    styles.stageIcon,
                    done && styles.stageIconDone,
                    active && styles.stageIconActive,
                  ]}
                >
                  {done ? (
                    <Ionicons name="checkmark" size={16} color={colors.background} />
                  ) : (
                    <Ionicons
                      name={stage.icon}
                      size={16}
                      color={active ? colors.background : colors.textDim}
                    />
                  )}
                </View>
                <Text
                  style={[
                    styles.stageLabel,
                    done && styles.stageLabelDone,
                    active && styles.stageLabelActive,
                  ]}
                >
                  {stage.label}
                </Text>
                {active && (
                  <View style={styles.activeDot} />
                )}
              </View>
            );
          })}
        </View>

        <Text style={styles.fieldsHint}>
          Extracting: {fields.join(' • ')}
        </Text>
      </View>
    </SafeAreaView>
  );
}

const colors = {
  background: '#0a0a12',
  surface: '#12121e',
  accent: '#00e5ff',
  accentGlow: '#00e5ff22',
  text: '#e8e8f0',
  textDim: '#666680',
  border: '#1e1e2e',
  success: '#00ff88',
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
  },
  orb: {
    width: 110,
    height: 110,
    borderRadius: 55,
    backgroundColor: colors.accentGlow,
    borderWidth: 1.5,
    borderColor: colors.accent + '66',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 32,
  },
  processingTitle: {
    fontSize: 26,
    fontWeight: '800',
    color: colors.text,
    marginBottom: 8,
  },
  processingSubtitle: {
    fontSize: 13,
    color: colors.textDim,
    marginBottom: 40,
    letterSpacing: 0.5,
  },
  stageList: {
    width: '100%',
    gap: 14,
    marginBottom: 40,
  },
  stageRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  stageIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stageIconDone: {
    backgroundColor: colors.success,
    borderColor: colors.success,
  },
  stageIconActive: {
    backgroundColor: colors.accent,
    borderColor: colors.accent,
  },
  stageLabel: {
    color: colors.textDim,
    fontSize: 14,
    fontWeight: '600',
    flex: 1,
  },
  stageLabelDone: {
    color: colors.success,
  },
  stageLabelActive: {
    color: colors.text,
    fontWeight: '700',
  },
  activeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.accent,
  },
  fieldsHint: {
    fontSize: 11,
    color: colors.textDim,
    textAlign: 'center',
    letterSpacing: 0.5,
  },
  errorBox: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
    gap: 16,
  },
  errorTitle: {
    fontSize: 22,
    fontWeight: '800',
    color: '#ff4444',
  },
  errorMsg: {
    fontSize: 14,
    color: colors.textDim,
    textAlign: 'center',
    lineHeight: 22,
  },
  retryBtn: {
    backgroundColor: colors.accent,
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 12,
    marginTop: 8,
  },
  retryBtnText: {
    color: colors.background,
    fontSize: 16,
    fontWeight: '800',
  },
  backLink: {
    padding: 8,
  },
  backLinkText: {
    color: colors.textDim,
    fontSize: 14,
  },
});
