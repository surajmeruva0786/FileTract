import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ScrollView,
  Image,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform,
  StatusBar,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { getDefaultFields, getPipeline } from '../services/storage';

const PRESET_FIELDS = {
  'Aadhaar Card': ['Name', 'Aadhaar Number', 'Date of Birth', 'Address', 'Gender'],
  'PAN Card': ['Name', 'Father Name', 'PAN Number', 'Date of Birth'],
  'Voter ID': ['Name', 'Father Name', 'Voter ID Number', 'Address', 'Date of Birth'],
  'Student ID': ['Name', 'Roll Number', 'Department', 'Year', 'Expiry Date'],
  'Driver License': ['Name', 'License Number', 'Date of Birth', 'Expiry Date', 'Address'],
  Custom: [],
};

export default function FieldsScreen({ navigation, route }) {
  const { image } = route.params;
  const [fields, setFields] = useState([]);
  const [newField, setNewField] = useState('');
  const [selectedPreset, setSelectedPreset] = useState(null);
  const [pipeline, setPipeline] = useState('standard');
  const inputRef = useRef(null);

  useEffect(() => {
    (async () => {
      const defaultFields = await getDefaultFields();
      const savedPipeline = await getPipeline();
      setFields(defaultFields);
      setPipeline(savedPipeline);
    })();
  }, []);

  const applyPreset = (presetName) => {
    setSelectedPreset(presetName);
    if (presetName !== 'Custom') {
      setFields([...PRESET_FIELDS[presetName]]);
    }
  };

  const addField = () => {
    const trimmed = newField.trim();
    if (trimmed && !fields.includes(trimmed)) {
      setFields((prev) => [...prev, trimmed]);
      setNewField('');
    }
  };

  const removeField = (field) => {
    setFields((prev) => prev.filter((f) => f !== field));
  };

  const proceed = () => {
    if (fields.length === 0) {
      alert('Please add at least one field to extract.');
      return;
    }
    navigation.navigate('Processing', { image, fields, pipeline });
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={colors.background} />
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView showsVerticalScrollIndicator={false}>
          {/* Header */}
          <View style={styles.headerRow}>
            <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
              <Ionicons name="arrow-back" size={24} color={colors.accent} />
            </TouchableOpacity>
            <Text style={styles.title}>Configure Extraction</Text>
          </View>

          {/* Image Preview */}
          <View style={styles.imageCard}>
            <Image source={{ uri: image.uri }} style={styles.imagePreview} resizeMode="cover" />
            <View style={styles.imageMeta}>
              <Ionicons name="document" size={16} color={colors.accent} />
              <Text style={styles.imageMetaText} numberOfLines={1}>
                {image.fileName || 'Captured Image'}
              </Text>
            </View>
          </View>

          {/* Preset Selector */}
          <Text style={styles.sectionLabel}>Document Type</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.presetScroll}>
            {Object.keys(PRESET_FIELDS).map((preset) => (
              <TouchableOpacity
                key={preset}
                style={[styles.presetChip, selectedPreset === preset && styles.presetChipActive]}
                onPress={() => applyPreset(preset)}
              >
                <Text
                  style={[
                    styles.presetChipText,
                    selectedPreset === preset && styles.presetChipTextActive,
                  ]}
                >
                  {preset}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>

          {/* Fields to Extract */}
          <Text style={styles.sectionLabel}>Fields to Extract</Text>
          <View style={styles.fieldsContainer}>
            {fields.map((field) => (
              <View key={field} style={styles.fieldChip}>
                <Text style={styles.fieldChipText}>{field}</Text>
                <TouchableOpacity onPress={() => removeField(field)}>
                  <Ionicons name="close-circle" size={18} color={colors.textDim} />
                </TouchableOpacity>
              </View>
            ))}
          </View>

          {/* Add Custom Field */}
          <View style={styles.addFieldRow}>
            <TextInput
              ref={inputRef}
              style={styles.fieldInput}
              placeholder="Add custom field (e.g. ID Number)"
              placeholderTextColor={colors.textDim}
              value={newField}
              onChangeText={setNewField}
              onSubmitEditing={addField}
              returnKeyType="done"
            />
            <TouchableOpacity style={styles.addBtn} onPress={addField}>
              <Ionicons name="add" size={24} color={colors.background} />
            </TouchableOpacity>
          </View>

          {/* Pipeline Selection */}
          <Text style={styles.sectionLabel}>Processing Pipeline</Text>
          <View style={styles.pipelineRow}>
            <TouchableOpacity
              style={[styles.pipelineBtn, pipeline === 'standard' && styles.pipelineBtnActive]}
              onPress={() => setPipeline('standard')}
            >
              <Ionicons
                name="flash"
                size={18}
                color={pipeline === 'standard' ? colors.background : colors.textDim}
              />
              <View>
                <Text
                  style={[
                    styles.pipelineBtnLabel,
                    pipeline === 'standard' && styles.pipelineBtnLabelActive,
                  ]}
                >
                  Standard
                </Text>
                <Text style={styles.pipelineBtnSub}>Fast • ~5 sec</Text>
              </View>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.pipelineBtn, pipeline === 'patent' && styles.pipelineBtnActive]}
              onPress={() => setPipeline('patent')}
            >
              <Ionicons
                name="shield-checkmark"
                size={18}
                color={pipeline === 'patent' ? colors.background : colors.accent}
              />
              <View>
                <Text
                  style={[
                    styles.pipelineBtnLabel,
                    pipeline === 'patent' && styles.pipelineBtnLabelActive,
                  ]}
                >
                  Patent
                </Text>
                <Text style={styles.pipelineBtnSub}>Accurate • ~30 sec</Text>
              </View>
            </TouchableOpacity>
          </View>

          {/* Extract Button */}
          <TouchableOpacity style={styles.extractBtn} onPress={proceed}>
            <Ionicons name="scan" size={22} color={colors.background} />
            <Text style={styles.extractBtnText}>Extract Fields</Text>
          </TouchableOpacity>

          <View style={{ height: 40 }} />
        </ScrollView>
      </KeyboardAvoidingView>
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
  backBtn: {
    padding: 4,
  },
  title: {
    fontSize: 22,
    fontWeight: '800',
    color: colors.text,
  },
  imageCard: {
    marginHorizontal: 20,
    marginTop: 16,
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  imagePreview: {
    width: '100%',
    height: 180,
  },
  imageMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 10,
  },
  imageMetaText: {
    color: colors.textDim,
    fontSize: 12,
    flex: 1,
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
  presetScroll: {
    paddingHorizontal: 20,
    marginBottom: 4,
  },
  presetChip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: colors.border,
    marginRight: 8,
    backgroundColor: colors.surface,
  },
  presetChipActive: {
    borderColor: colors.accent,
    backgroundColor: colors.accentGlow,
  },
  presetChipText: {
    color: colors.textDim,
    fontSize: 13,
    fontWeight: '600',
  },
  presetChipTextActive: {
    color: colors.accent,
  },
  fieldsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    paddingHorizontal: 20,
    marginBottom: 4,
  },
  fieldChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  fieldChipText: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '600',
  },
  addFieldRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: 20,
    marginTop: 12,
    gap: 10,
  },
  fieldInput: {
    flex: 1,
    backgroundColor: colors.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    color: colors.text,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 14,
  },
  addBtn: {
    backgroundColor: colors.accent,
    borderRadius: 12,
    padding: 12,
  },
  pipelineRow: {
    flexDirection: 'row',
    gap: 12,
    paddingHorizontal: 20,
    marginBottom: 4,
  },
  pipelineBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    padding: 16,
    borderRadius: 14,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  pipelineBtnActive: {
    backgroundColor: colors.accent,
    borderColor: colors.accent,
  },
  pipelineBtnLabel: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '700',
  },
  pipelineBtnLabelActive: {
    color: colors.background,
  },
  pipelineBtnSub: {
    color: colors.textDim,
    fontSize: 11,
    marginTop: 2,
  },
  extractBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    backgroundColor: colors.accent,
    marginHorizontal: 20,
    marginTop: 28,
    paddingVertical: 18,
    borderRadius: 14,
  },
  extractBtnText: {
    color: colors.background,
    fontSize: 17,
    fontWeight: '800',
  },
});
