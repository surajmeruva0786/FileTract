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
import { sendToGoogleSheets } from '../services/googleSheets';

function parseResults(results, pipeline) {
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

const QUALITY_COLORS = {
  reliable: '#00ff88',
  good: '#00e5ff',
  uncertain: '#ffaa00',
  'low-quality': '#ff4444',
};

export default function PreviewScreen({ navigation, route }) {
  const { image, fields, results, pipeline } = route.params;
  const [extractedFields, setExtractedFields] = useState(() => parseResults(results, pipeline));
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
      Alert.alert(
        'Exported!',
        'Data has been successfully added to your Google Sheet.',
        [{ text: 'OK' }]
      );
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
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={colors.background} />

      {/* Header */}
      <View style={styles.headerRow}>
        <TouchableOpacity onPress={() => navigation.navigate('Home')} style={styles.backBtn}>
          <Ionicons name="home" size={22} color={colors.accent} />
        </TouchableOpacity>
        <Text style={styles.title}>Extraction Results</Text>
        <View style={styles.successBadge}>
          <Ionicons name="checkmark-circle" size={14} color={colors.success} />
          <Text style={styles.successBadgeText}>Complete</Text>
        </View>
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Image Thumbnail */}
        <View style={styles.imageRow}>
          <Image source={{ uri: image.uri }} style={styles.thumbnail} resizeMode="cover" />
          <View style={styles.imageMeta}>
            <Text style={styles.imageLabel}>{image.fileName || 'ID Card'}</Text>
            <Text style={styles.pipelineTag}>
              {pipeline === 'patent' ? 'Patent Pipeline' : 'Standard Pipeline'}
            </Text>
            {qualityReport && (
              <Text style={styles.qualityLine}>
                Overall: {qualityReport.overall_quality} •{' '}
                {qualityReport.reliable_fields}/{qualityReport.total_fields} reliable
              </Text>
            )}
          </View>
        </View>

        {/* Extracted Fields */}
        <Text style={styles.sectionLabel}>Extracted Fields</Text>
        <View style={styles.fieldsCard}>
          {extractedFields.length === 0 ? (
            <Text style={styles.emptyText}>No fields extracted. Try again with a clearer image.</Text>
          ) : (
            extractedFields.map(({ field, value, confidence, quality }, idx) => (
              <View key={field} style={styles.fieldRow}>
                <View style={styles.fieldMeta}>
                  <Text style={styles.fieldLabel}>{field}</Text>
                  {quality && (
                    <View
                      style={[
                        styles.qualityDot,
                        { backgroundColor: QUALITY_COLORS[quality] || colors.textDim },
                      ]}
                    />
                  )}
                </View>

                {editingIdx === idx ? (
                  <View style={styles.editRow}>
                    <TextInput
                      style={styles.editInput}
                      value={editValue}
                      onChangeText={setEditValue}
                      autoFocus
                      returnKeyType="done"
                      onSubmitEditing={saveEdit}
                    />
                    <TouchableOpacity style={styles.saveBtn} onPress={saveEdit}>
                      <Ionicons name="checkmark" size={18} color={colors.background} />
                    </TouchableOpacity>
                  </View>
                ) : (
                  <TouchableOpacity style={styles.valueRow} onLongPress={() => startEdit(idx)}>
                    <Text style={styles.fieldValue}>{value || '—'}</Text>
                    <TouchableOpacity onPress={() => startEdit(idx)}>
                      <Ionicons name="pencil" size={16} color={colors.textDim} />
                    </TouchableOpacity>
                  </TouchableOpacity>
                )}

                {confidence !== null && (
                  <View style={styles.confBar}>
                    <View style={[styles.confFill, { width: `${Math.round(confidence * 100)}%` }]} />
                    <Text style={styles.confText}>{Math.round(confidence * 100)}% OCR</Text>
                  </View>
                )}

                {idx < extractedFields.length - 1 && <View style={styles.divider} />}
              </View>
            ))
          )}
        </View>

        {/* Export Actions */}
        <Text style={styles.sectionLabel}>Export</Text>

        <TouchableOpacity
          style={[styles.exportBtn, sent && styles.exportBtnDone]}
          onPress={handleExportToSheets}
          disabled={sending || sent}
        >
          {sending ? (
            <ActivityIndicator color={colors.background} />
          ) : sent ? (
            <Ionicons name="checkmark-circle" size={22} color={colors.background} />
          ) : (
            <Ionicons name="logo-google" size={22} color={colors.background} />
          )}
          <Text style={styles.exportBtnText}>
            {sent ? 'Sent to Google Sheets' : 'Export to Google Sheets'}
          </Text>
        </TouchableOpacity>

        <Text style={styles.editHint}>Tip: Long-press or tap the pencil icon to edit any field before exporting.</Text>

        <TouchableOpacity
          style={styles.newScanBtn}
          onPress={() => navigation.navigate('Home')}
        >
          <Ionicons name="scan" size={20} color={colors.accent} />
          <Text style={styles.newScanText}>Scan Another Document</Text>
        </TouchableOpacity>

        <View style={{ height: 40 }} />
      </ScrollView>
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
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 10,
    gap: 12,
  },
  backBtn: { padding: 4 },
  title: {
    fontSize: 20,
    fontWeight: '800',
    color: colors.text,
    flex: 1,
  },
  successBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: colors.success + '22',
    borderWidth: 1,
    borderColor: colors.success + '44',
  },
  successBadgeText: {
    color: colors.success,
    fontSize: 11,
    fontWeight: '700',
  },
  imageRow: {
    flexDirection: 'row',
    gap: 14,
    marginHorizontal: 20,
    marginTop: 16,
    padding: 14,
    backgroundColor: colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
  },
  thumbnail: {
    width: 80,
    height: 60,
    borderRadius: 8,
  },
  imageMeta: {
    flex: 1,
    justifyContent: 'center',
    gap: 4,
  },
  imageLabel: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '700',
  },
  pipelineTag: {
    color: colors.accent,
    fontSize: 11,
    fontWeight: '600',
  },
  qualityLine: {
    color: colors.textDim,
    fontSize: 11,
  },
  sectionLabel: {
    color: colors.textDim,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.5,
    textTransform: 'uppercase',
    marginTop: 24,
    marginBottom: 10,
    paddingHorizontal: 20,
  },
  fieldsCard: {
    marginHorizontal: 20,
    backgroundColor: colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  fieldRow: {
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  fieldMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
  },
  fieldLabel: {
    color: colors.textDim,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  qualityDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  valueRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  fieldValue: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  editRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  editInput: {
    flex: 1,
    backgroundColor: colors.background,
    color: colors.text,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.accent,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 15,
  },
  saveBtn: {
    backgroundColor: colors.accent,
    borderRadius: 8,
    padding: 8,
  },
  confBar: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
    height: 4,
    backgroundColor: colors.border,
    borderRadius: 2,
    gap: 8,
    overflow: 'hidden',
  },
  confFill: {
    height: 4,
    backgroundColor: colors.accent + '88',
    borderRadius: 2,
  },
  confText: {
    color: colors.textDim,
    fontSize: 10,
    position: 'absolute',
    right: 0,
  },
  divider: {
    height: 1,
    backgroundColor: colors.border,
    marginTop: 14,
  },
  emptyText: {
    color: colors.textDim,
    fontSize: 14,
    textAlign: 'center',
    padding: 24,
  },
  exportBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    backgroundColor: '#4CAF50',
    marginHorizontal: 20,
    paddingVertical: 18,
    borderRadius: 14,
  },
  exportBtnDone: {
    backgroundColor: colors.success,
  },
  exportBtnText: {
    color: colors.background,
    fontSize: 16,
    fontWeight: '800',
  },
  editHint: {
    color: colors.textDim,
    fontSize: 11,
    textAlign: 'center',
    marginTop: 10,
    paddingHorizontal: 20,
  },
  newScanBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    marginHorizontal: 20,
    marginTop: 16,
    paddingVertical: 16,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: colors.accent,
  },
  newScanText: {
    color: colors.accent,
    fontSize: 15,
    fontWeight: '700',
  },
});
