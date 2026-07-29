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
import { LinearGradient } from 'expo-linear-gradient';
import { getDefaultFields, getPipeline } from '../services/storage';
import GlowBackground from '../components/GlowBackground';
import { colors, gradients, fonts } from '../theme';

const PRESETS = [
  { name: 'Aadhaar Card', icon: 'card', fields: ['Name', 'Aadhaar Number', 'Date of Birth', 'Address', 'Gender'] },
  { name: 'PAN Card', icon: 'card-outline', fields: ['Name', 'Father Name', 'PAN Number', 'Date of Birth'] },
  { name: 'Voter ID', icon: 'checkbox-outline', fields: ['Name', 'Father Name', 'Voter ID Number', 'Address', 'Date of Birth'] },
  { name: 'Student ID', icon: 'school-outline', fields: ['Name', 'Roll Number', 'Department', 'Year', 'Expiry Date'] },
  { name: 'Driver License', icon: 'car-outline', fields: ['Name', 'License Number', 'Date of Birth', 'Expiry Date', 'Address'] },
  { name: 'Custom', icon: 'options-outline', fields: [] },
];

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

  const applyPreset = (preset) => {
    setSelectedPreset(preset.name);
    if (preset.name !== 'Custom') {
      setFields([...preset.fields]);
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

  const canExtract = fields.length > 0;

  return (
    <View style={styles.root}>
      <GlowBackground />
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor={colors.background} />
        <KeyboardAvoidingView
          style={{ flex: 1 }}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        >
          <ScrollView showsVerticalScrollIndicator={false}>
            <View style={styles.headerRow}>
              <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
                <Ionicons name="arrow-back" size={20} color="#fff" />
              </TouchableOpacity>
              <Text style={styles.title}>Configure Extraction</Text>
            </View>

            <View style={styles.imageCard}>
              <Image source={{ uri: image.uri }} style={styles.imagePreview} resizeMode="cover" />
              <View style={styles.imageMeta}>
                <Ionicons name="document" size={14} color={colors.violet} />
                <Text style={styles.imageMetaText} numberOfLines={1}>
                  {image.fileName || 'Captured Image'}
                </Text>
              </View>
            </View>

            <Text style={styles.sectionLabel}>Document Type</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.presetScroll}>
              {PRESETS.map((preset) => {
                const active = selectedPreset === preset.name;
                return (
                  <TouchableOpacity key={preset.name} onPress={() => applyPreset(preset)} activeOpacity={0.85}>
                    {active ? (
                      <LinearGradient colors={gradients.primary} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.presetChip}>
                        <Ionicons name={preset.icon} size={15} color="#fff" />
                        <Text style={[styles.presetChipText, styles.presetChipTextActive]}>{preset.name}</Text>
                      </LinearGradient>
                    ) : (
                      <View style={[styles.presetChip, styles.presetChipInactive]}>
                        <Ionicons name={preset.icon} size={15} color="rgba(255,255,255,0.65)" />
                        <Text style={styles.presetChipText}>{preset.name}</Text>
                      </View>
                    )}
                  </TouchableOpacity>
                );
              })}
            </ScrollView>

            <Text style={styles.sectionLabel}>Fields to Extract</Text>
            <View style={styles.fieldsContainer}>
              {fields.map((field) => (
                <View key={field} style={styles.fieldChip}>
                  <Text style={styles.fieldChipText}>{field}</Text>
                  <TouchableOpacity onPress={() => removeField(field)}>
                    <Ionicons name="close" size={13} color="rgba(255,255,255,0.6)" />
                  </TouchableOpacity>
                </View>
              ))}
            </View>

            <View style={styles.addFieldRow}>
              <TextInput
                ref={inputRef}
                style={styles.fieldInput}
                placeholder="Add a field…"
                placeholderTextColor={colors.textGhost}
                value={newField}
                onChangeText={setNewField}
                onSubmitEditing={addField}
                returnKeyType="done"
              />
              <TouchableOpacity onPress={addField}>
                <LinearGradient colors={gradients.primary} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.addBtn}>
                  <Ionicons name="add" size={20} color="#fff" />
                </LinearGradient>
              </TouchableOpacity>
            </View>

            <Text style={styles.sectionLabel}>Processing Mode</Text>
            <View style={styles.segmentRow}>
              <TouchableOpacity style={styles.segmentTouchable} onPress={() => setPipeline('standard')}>
                {pipeline === 'standard' ? (
                  <LinearGradient colors={gradients.primary} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.segmentBtn}>
                    <Ionicons name="flash" size={15} color="#fff" />
                    <Text style={styles.segmentTextActive}>Fast</Text>
                  </LinearGradient>
                ) : (
                  <View style={styles.segmentBtn}>
                    <Ionicons name="flash-outline" size={15} color="rgba(255,255,255,0.5)" />
                    <Text style={styles.segmentText}>Fast</Text>
                  </View>
                )}
              </TouchableOpacity>
              <TouchableOpacity style={styles.segmentTouchable} onPress={() => setPipeline('patent')}>
                {pipeline === 'patent' ? (
                  <LinearGradient colors={gradients.primary} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.segmentBtn}>
                    <Ionicons name="shield-checkmark" size={15} color="#fff" />
                    <Text style={styles.segmentTextActive}>Accurate</Text>
                  </LinearGradient>
                ) : (
                  <View style={styles.segmentBtn}>
                    <Ionicons name="shield-checkmark-outline" size={15} color="rgba(255,255,255,0.5)" />
                    <Text style={styles.segmentText}>Accurate</Text>
                  </View>
                )}
              </TouchableOpacity>
            </View>
            <Text style={styles.modeHelper}>
              {pipeline === 'standard'
                ? 'Optimized for speed on straightforward documents.'
                : 'A deeper verification pass for critical or complex documents.'}
            </Text>

            <TouchableOpacity activeOpacity={0.85} onPress={proceed} disabled={!canExtract} style={{ marginTop: 28 }}>
              {canExtract ? (
                <LinearGradient colors={gradients.primary} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.extractBtn}>
                  <Text style={styles.extractBtnText}>Extract Fields</Text>
                  <Ionicons name="arrow-forward" size={17} color="#fff" />
                </LinearGradient>
              ) : (
                <View style={[styles.extractBtn, styles.extractBtnDisabled]}>
                  <Text style={styles.extractBtnTextDisabled}>Extract Fields</Text>
                  <Ionicons name="arrow-forward" size={17} color="rgba(255,255,255,0.35)" />
                </View>
              )}
            </TouchableOpacity>

            <View style={{ height: 40 }} />
          </ScrollView>
        </KeyboardAvoidingView>
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
    paddingBottom: 8,
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
  imageCard: {
    marginHorizontal: 22,
    marginTop: 16,
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  imagePreview: {
    width: '100%',
    height: 170,
  },
  imageMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 10,
  },
  imageMetaText: {
    fontFamily: fonts.body,
    color: colors.textDim,
    fontSize: 12,
    flex: 1,
  },
  sectionLabel: {
    fontFamily: fonts.bodyBold,
    color: colors.textFaint,
    fontSize: 11,
    letterSpacing: 1.5,
    textTransform: 'uppercase',
    marginTop: 26,
    marginBottom: 12,
    paddingHorizontal: 22,
  },
  presetScroll: {
    paddingHorizontal: 22,
  },
  presetChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 999,
    marginRight: 10,
  },
  presetChipInactive: {
    backgroundColor: colors.chip,
    borderWidth: 1,
    borderColor: colors.border,
  },
  presetChipText: {
    fontFamily: fonts.bodySemi,
    fontSize: 13.5,
    color: 'rgba(255,255,255,0.65)',
  },
  presetChipTextActive: {
    color: '#fff',
  },
  fieldsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 9,
    paddingHorizontal: 22,
  },
  fieldChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingLeft: 14,
    paddingRight: 8,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: colors.chip,
    borderWidth: 1,
    borderColor: colors.border,
  },
  fieldChipText: {
    fontFamily: fonts.body,
    color: 'rgba(255,255,255,0.85)',
    fontSize: 13.5,
  },
  addFieldRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: 22,
    marginTop: 12,
    gap: 9,
  },
  fieldInput: {
    flex: 1,
    fontFamily: fonts.body,
    backgroundColor: colors.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    color: colors.text,
    paddingHorizontal: 14,
    paddingVertical: 11,
    fontSize: 13.5,
  },
  addBtn: {
    width: 42,
    height: 42,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  segmentRow: {
    flexDirection: 'row',
    gap: 4,
    marginHorizontal: 22,
    padding: 4,
    borderRadius: 999,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderWidth: 1,
    borderColor: colors.border,
  },
  segmentTouchable: {
    flex: 1,
  },
  segmentBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    paddingVertical: 10,
    borderRadius: 999,
  },
  segmentText: {
    fontFamily: fonts.bodySemi,
    fontSize: 13.5,
    color: 'rgba(255,255,255,0.5)',
  },
  segmentTextActive: {
    fontFamily: fonts.bodySemi,
    fontSize: 13.5,
    color: '#fff',
  },
  modeHelper: {
    fontFamily: fonts.body,
    marginHorizontal: 22,
    marginTop: 10,
    fontSize: 12.5,
    lineHeight: 18,
    color: colors.textFaint,
  },
  extractBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 9,
    marginHorizontal: 22,
    paddingVertical: 16,
    borderRadius: 16,
  },
  extractBtnDisabled: {
    backgroundColor: 'rgba(255,255,255,0.06)',
  },
  extractBtnText: {
    fontFamily: fonts.bodySemi,
    color: '#fff',
    fontSize: 15.5,
  },
  extractBtnTextDisabled: {
    fontFamily: fonts.bodySemi,
    color: 'rgba(255,255,255,0.35)',
    fontSize: 15.5,
  },
});
