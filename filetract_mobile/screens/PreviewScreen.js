import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Image,
  SafeAreaView,
  StatusBar,
  TextInput,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { sendToGoogleSheets } from '../services/googleSheets';
import GlowBackground from '../components/GlowBackground';
import { colors, gradients, fonts, qualityBadge } from '../theme';

function parseResults(results) {
  // Patent pipeline format
  if (results?.results?.extracted_fields) {
    return Object.entries(results.results.extracted_fields).map(([field, data]) => ({
      field,
      value: typeof data === 'object' ? (data.value ?? '') : String(data),
      confidence: typeof data === 'object' ? data.ocr_confidence : null,
      quality: typeof data === 'object' ? data.quality_flag : null,
    }));
  }
  // Standard pipeline format
  if (results?.results) {
    return Object.entries(results.results).map(([field, value]) => ({
      field,
      value: String(value ?? ''),
      confidence: null,
      quality: null,
    }));
  }
  return [];
}

export default function PreviewScreen({ navigation, route }) {
  const { image, fields, results, pipeline } = route.params;
  const [extractedFields, setExtractedFields] = useState(() => parseResults(results));
  const [editingIdx, setEditingIdx] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);

  const startEdit = (idx) => {
    setEditingIdx(idx);
    setEditValue(extractedFields[idx].value);
  };

  const saveEdit = () => {
    if (editingIdx === null) return;
    setExtractedFields((prev) =>
      prev.map((f, i) => (i === editingIdx ? { ...f, value: editValue } : f))
    );
    setEditingIdx(null);
  };

  const handleExportToSheets = useCallback(async () => {
    setSending(true);
    try {
      const flatData = {};
      extractedFields.forEach(({ field, value }) => {
        flatData[field] = value;
      });
      await sendToGoogleSheets({ results: flatData }, image.fileName || 'id_card.jpg');
      setSent(true);
      setTimeout(() => setSent(false), 2600);
    } catch (err) {
      Alert.alert('Export Failed', err.message, [
        { text: 'Go to Settings', onPress: () => navigation.navigate('Settings') },
        { text: 'Cancel', style: 'cancel' },
      ]);
    } finally {
      setSending(false);
    }
  }, [extractedFields, image]);

  const qualityReport = results?.results?.quality_report || null;

  return (
    <View style={styles.root}>
      <GlowBackground />
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor={colors.background} />

        <View style={styles.headerRow}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
            <Ionicons name="arrow-back" size={20} color="#fff" />
          </TouchableOpacity>
          <View style={{ flex: 1 }}>
            <Text style={styles.title}>Extracted Fields</Text>
            <Text style={styles.subtitle}>
              {pipeline === 'patent' ? 'Patent Pipeline' : 'Standard Pipeline'} · Just now
            </Text>
          </View>
          <LinearGradient colors={gradients.orb} style={styles.headerIcon}>
            <Ionicons name="document-text" size={22} color="#fff" />
          </LinearGradient>
        </View>

        <ScrollView showsVerticalScrollIndicator={false}>
          <View style={styles.imageRow}>
            <Image source={{ uri: image.uri }} style={styles.thumbnail} resizeMode="cover" />
            <View style={styles.imageMeta}>
              <Text style={styles.imageLabel} numberOfLines={1}>{image.fileName || 'ID Card'}</Text>
              {qualityReport && (
                <Text style={styles.qualityLine}>
                  Overall: {qualityReport.overall_quality} · {qualityReport.reliable_fields}/{qualityReport.total_fields} reliable
                </Text>
              )}
            </View>
          </View>

          <View style={styles.fieldsCard}>
            {extractedFields.length === 0 ? (
              <Text style={styles.emptyText}>No fields extracted. Try again with a clearer image.</Text>
            ) : (
              extractedFields.map(({ field, value, confidence, quality }, idx) => {
                const badge = qualityBadge(quality);
                return (
                  <View key={field} style={[styles.fieldRow, idx > 0 && styles.fieldRowDivider]}>
                    <View style={styles.fieldMetaRow}>
                      <Text style={styles.fieldLabel}>{field}</Text>
                      {badge && (
                        <View style={[styles.badge, { backgroundColor: badge.bg }]}>
                          <Text style={[styles.badgeText, { color: badge.color }]}>{badge.label}</Text>
                        </View>
                      )}
                    </View>

                    {editingIdx === idx ? (
                      <TextInput
                        style={styles.editInput}
                        value={editValue}
                        onChangeText={setEditValue}
                        autoFocus
                        returnKeyType="done"
                        onSubmitEditing={saveEdit}
                        onBlur={saveEdit}
                      />
                    ) : (
                      <TouchableOpacity style={styles.valueRow} onPress={() => startEdit(idx)}>
                        <Text style={styles.fieldValue}>{value || '—'}</Text>
                        <Ionicons name="pencil-outline" size={14} color="rgba(255,255,255,0.3)" />
                      </TouchableOpacity>
                    )}

                    {confidence !== null && (
                      <View style={styles.confBar}>
                        <View style={[styles.confFill, { width: `${Math.round(confidence * 100)}%` }]} />
                        <Text style={styles.confText}>{Math.round(confidence * 100)}% OCR</Text>
                      </View>
                    )}
                  </View>
                );
              })
            )}
          </View>

          <TouchableOpacity activeOpacity={0.85} onPress={handleExportToSheets} disabled={sending} style={styles.exportWrap}>
            <LinearGradient
              colors={sent ? gradients.success : gradients.primary}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.exportBtn}
            >
              {sending ? (
                <ActivityIndicator color="#fff" />
              ) : sent ? (
                <Ionicons name="checkmark" size={18} color="#fff" />
              ) : (
                <Ionicons name="logo-google" size={18} color="#fff" />
              )}
              <Text style={styles.exportBtnText}>
                {sending ? 'Sending…' : sent ? 'Sent to Google Sheets' : 'Send to Google Sheets'}
              </Text>
            </LinearGradient>
          </TouchableOpacity>

          <Text style={styles.editHint}>Tap any field to edit its value before exporting.</Text>

          <TouchableOpacity style={styles.newScanBtn} onPress={() => navigation.navigate('Home')}>
            <Ionicons name="scan-outline" size={17} color={colors.violet} />
            <Text style={styles.newScanText}>Scan Another Document</Text>
          </TouchableOpacity>

          <View style={{ height: 40 }} />
        </ScrollView>
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
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 22,
    paddingTop: 22,
    paddingBottom: 18,
    gap: 14,
  },
  backBtn: {
    width: 38,
    height: 38,
    borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontFamily: fonts.displaySemi,
    fontSize: 19,
    color: colors.text,
  },
  subtitle: {
    fontFamily: fonts.body,
    fontSize: 12,
    color: colors.textFaint,
    marginTop: 2,
  },
  headerIcon: {
    width: 46,
    height: 46,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
  },
  imageRow: {
    flexDirection: 'row',
    gap: 14,
    marginHorizontal: 22,
    padding: 14,
    backgroundColor: colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
  },
  thumbnail: {
    width: 64,
    height: 48,
    borderRadius: 8,
  },
  imageMeta: {
    flex: 1,
    gap: 4,
  },
  imageLabel: {
    fontFamily: fonts.bodyBold,
    color: colors.text,
    fontSize: 13,
  },
  qualityLine: {
    fontFamily: fonts.body,
    color: colors.textFaint,
    fontSize: 11,
  },
  fieldsCard: {
    marginHorizontal: 22,
    marginTop: 20,
    gap: 10,
  },
  fieldRow: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 16,
    padding: 14,
  },
  fieldRowDivider: {},
  fieldMetaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  fieldLabel: {
    fontFamily: fonts.bodySemi,
    fontSize: 12,
    color: colors.textFaint,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderRadius: 999,
  },
  badgeText: {
    fontFamily: fonts.bodyBold,
    fontSize: 10.5,
    textTransform: 'uppercase',
    letterSpacing: 0.4,
  },
  valueRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  fieldValue: {
    fontFamily: fonts.bodyMedium,
    color: colors.text,
    fontSize: 15.5,
    flex: 1,
  },
  editInput: {
    fontFamily: fonts.body,
    backgroundColor: '#1F1832',
    borderWidth: 1,
    borderColor: colors.violet,
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
    color: colors.text,
    fontSize: 15,
  },
  confBar: {
    marginTop: 10,
    height: 4,
    backgroundColor: colors.border,
    borderRadius: 2,
    overflow: 'hidden',
  },
  confFill: {
    height: 4,
    backgroundColor: colors.violet,
    borderRadius: 2,
  },
  confText: {
    fontFamily: fonts.body,
    color: colors.textFaint,
    fontSize: 10,
    marginTop: 4,
    textAlign: 'right',
  },
  emptyText: {
    fontFamily: fonts.body,
    color: colors.textDim,
    fontSize: 14,
    textAlign: 'center',
    padding: 24,
  },
  exportWrap: {
    marginHorizontal: 22,
    marginTop: 22,
  },
  exportBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 16,
    borderRadius: 16,
  },
  exportBtnText: {
    fontFamily: fonts.bodySemi,
    color: '#fff',
    fontSize: 15.5,
  },
  editHint: {
    fontFamily: fonts.body,
    color: colors.textFaint,
    fontSize: 11,
    textAlign: 'center',
    marginTop: 10,
    paddingHorizontal: 20,
  },
  newScanBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 9,
    marginHorizontal: 22,
    marginTop: 16,
    paddingVertical: 15,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: colors.borderStrong,
  },
  newScanText: {
    fontFamily: fonts.bodySemi,
    color: colors.violet,
    fontSize: 14.5,
  },
});
