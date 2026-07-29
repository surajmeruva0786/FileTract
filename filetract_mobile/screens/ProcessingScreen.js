import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  Easing,
  SafeAreaView,
  StatusBar,
  TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { processImage, setBackendUrl } from '../services/api';
import { getBackendUrl as getStoredUrl } from '../services/storage';
import GlowBackground from '../components/GlowBackground';
import { colors, gradients, fonts } from '../theme';

// Real stage counts/semantics match app.py's actual pipelines (see CLAUDE.md
// 2026-07-01 SOTA v3.0 entry): patent = 5-stage SOTA pipeline, standard = 2 phases.
const ACCURATE_STAGES = [
  { id: 1, label: 'Preprocessing Image', icon: 'scan' },
  { id: 2, label: 'Detecting Document Type', icon: 'search' },
  { id: 3, label: 'Running OCR Extraction', icon: 'layers' },
  { id: 4, label: 'Cross-Checking Fields', icon: 'shield-checkmark' },
  { id: 5, label: 'Scoring Confidence', icon: 'analytics' },
];
const FAST_STAGES = [
  { id: 1, label: 'Uploading & Preprocessing', icon: 'cloud-upload' },
  { id: 2, label: 'Extracting Fields', icon: 'sparkles' },
];

export default function ProcessingScreen({ navigation, route }) {
  const { image, fields, pipeline } = route.params;
  const [currentStage, setCurrentStage] = useState(0);
  const [error, setError] = useState(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const spinAnim = useRef(new Animated.Value(0)).current;

  const stageList = pipeline === 'patent' ? ACCURATE_STAGES : FAST_STAGES;

  useEffect(() => {
    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.08, duration: 900, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 900, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
      ])
    );
    pulse.start();
    const spin = Animated.loop(
      Animated.timing(spinAnim, { toValue: 1, duration: 1400, easing: Easing.linear, useNativeDriver: true })
    );
    spin.start();
    return () => {
      pulse.stop();
      spin.stop();
    };
  }, []);

  useEffect(() => {
    run();
  }, []);

  const run = async () => {
    try {
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

  const spin = spinAnim.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });

  if (error) {
    return (
      <View style={styles.root}>
        <GlowBackground />
        <SafeAreaView style={styles.container}>
          <StatusBar barStyle="light-content" backgroundColor={colors.background} />
          <View style={styles.errorBox}>
            <View style={styles.errorIconWrap}>
              <Ionicons name="warning" size={36} color={colors.danger} />
            </View>
            <Text style={styles.errorTitle}>Extraction Failed</Text>
            <Text style={styles.errorMsg}>{error}</Text>
            <View style={styles.errorActions}>
              <TouchableOpacity activeOpacity={0.85} onPress={() => { setError(null); setCurrentStage(0); run(); }}>
                <LinearGradient colors={gradients.primary} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.retryBtn}>
                  <Ionicons name="refresh" size={16} color="#fff" />
                  <Text style={styles.retryBtnText}>Try Again</Text>
                </LinearGradient>
              </TouchableOpacity>
              <TouchableOpacity style={styles.backLink} onPress={() => navigation.goBack()}>
                <Text style={styles.backLinkText}>Back to Fields</Text>
              </TouchableOpacity>
            </View>
          </View>
        </SafeAreaView>
      </View>
    );
  }

  return (
    <View style={styles.root}>
      <GlowBackground />
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor={colors.background} />

        <View style={styles.center}>
          <Text style={styles.eyebrow}>Extracting Document</Text>

          <View style={styles.orbWrap}>
            <Animated.View style={[styles.spinRing, { transform: [{ rotate: spin }] }]} />
            <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
              <LinearGradient colors={gradients.orb} style={styles.orb}>
                <Ionicons name="scan" size={32} color="#fff" />
              </LinearGradient>
            </Animated.View>
          </View>

          <View style={styles.stageList}>
            {stageList.map((stage, idx) => {
              const done = currentStage > stage.id;
              const active = currentStage === stage.id;
              const isLast = idx === stageList.length - 1;
              return (
                <View key={stage.id} style={styles.stageRow}>
                  <View style={styles.stageIconCol}>
                    {done || active ? (
                      <LinearGradient
                        colors={done ? gradients.orb : gradients.primary}
                        style={styles.stageIcon}
                      >
                        <Ionicons name={done ? 'checkmark' : stage.icon} size={18} color="#fff" />
                      </LinearGradient>
                    ) : (
                      <View style={[styles.stageIcon, styles.stageIconPending]}>
                        <Ionicons name={stage.icon} size={17} color="rgba(255,255,255,0.35)" />
                      </View>
                    )}
                    {!isLast && (
                      <View style={[styles.connector, done && styles.connectorDone]} />
                    )}
                  </View>
                  <View style={styles.stageTextCol}>
                    <Text style={[styles.stageLabel, (done || active) && styles.stageLabelActive]}>
                      {stage.label}
                    </Text>
                    {active && <Text style={styles.stageStatus}>In progress…</Text>}
                    {done && <Text style={styles.stageStatus}>Done</Text>}
                  </View>
                </View>
              );
            })}
          </View>

          <Text style={styles.fieldsHint}>Extracting: {fields.join(' • ')}</Text>
        </View>
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
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
  },
  eyebrow: {
    fontFamily: fonts.bodyBold,
    fontSize: 11,
    letterSpacing: 2,
    textTransform: 'uppercase',
    color: colors.violet,
    marginBottom: 30,
  },
  orbWrap: {
    width: 96,
    height: 96,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 36,
  },
  spinRing: {
    position: 'absolute',
    width: 96,
    height: 96,
    borderRadius: 48,
    borderWidth: 2.5,
    borderColor: 'transparent',
    borderTopColor: colors.violet,
    borderRightColor: colors.violet,
  },
  orb: {
    width: 70,
    height: 70,
    borderRadius: 35,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stageList: {
    width: '100%',
    maxWidth: 320,
    marginBottom: 30,
  },
  stageRow: {
    flexDirection: 'row',
    gap: 16,
  },
  stageIconCol: {
    alignItems: 'center',
  },
  stageIcon: {
    width: 40,
    height: 40,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stageIconPending: {
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: 1,
    borderColor: colors.border,
  },
  connector: {
    width: 2,
    flex: 1,
    minHeight: 16,
    marginVertical: 2,
    backgroundColor: colors.border,
  },
  connectorDone: {
    backgroundColor: colors.violetDeep,
  },
  stageTextCol: {
    paddingTop: 8,
    paddingBottom: 18,
    flex: 1,
  },
  stageLabel: {
    fontFamily: fonts.bodySemi,
    fontSize: 14.5,
    color: 'rgba(255,255,255,0.4)',
  },
  stageLabelActive: {
    color: '#fff',
  },
  stageStatus: {
    fontFamily: fonts.body,
    fontSize: 12,
    color: 'rgba(167,139,250,0.75)',
    marginTop: 2,
  },
  fieldsHint: {
    fontFamily: fonts.body,
    fontSize: 11,
    color: colors.textFaint,
    textAlign: 'center',
    letterSpacing: 0.3,
  },
  errorBox: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
    gap: 16,
  },
  errorIconWrap: {
    width: 84,
    height: 84,
    borderRadius: 42,
    backgroundColor: 'rgba(239,68,68,0.12)',
    borderWidth: 1,
    borderColor: 'rgba(239,68,68,0.25)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorTitle: {
    fontFamily: fonts.display,
    fontSize: 20,
    color: colors.text,
  },
  errorMsg: {
    fontFamily: fonts.body,
    fontSize: 13.5,
    color: colors.textDim,
    textAlign: 'center',
    lineHeight: 22,
    maxWidth: 280,
  },
  errorActions: {
    width: '100%',
    maxWidth: 280,
    gap: 10,
    marginTop: 8,
  },
  retryBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 9,
    paddingVertical: 14,
    borderRadius: 14,
  },
  retryBtnText: {
    fontFamily: fonts.bodySemi,
    color: '#fff',
    fontSize: 14.5,
  },
  backLink: {
    padding: 12,
    alignItems: 'center',
  },
  backLinkText: {
    fontFamily: fonts.bodyMedium,
    color: colors.textDim,
    fontSize: 13.5,
  },
});
