import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  StatusBar,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import {
  saveBackendUrl,
  getBackendUrl,
  savePipeline,
  getPipeline,
  saveDefaultFields,
  getDefaultFields,
} from '../services/storage';
import { saveSheetsUrl, getSheetsUrl } from '../services/googleSheets';
import { setBackendUrl } from '../services/api';
import GlowBackground from '../components/GlowBackground';
import { colors, gradients, fonts } from '../theme';

const SETUP_STEPS = [
  'Open your Google Sheet.',
  'Go to Extensions → Apps Script.',
  'Paste the contents of filetract_mobile/google_apps_script/Code.gs.',
  'Click Deploy → New Deployment.',
  'Select type: Web App.',
  'Set "Execute as": Me, "Who has access": Anyone.',
  'Click Deploy and copy the Web App URL.',
  'Paste it in the field above and save.',
];

export default function SettingsScreen({ navigation }) {
  const [backendUrl, setBackendUrlState] = useState('');
  const [sheetsUrl, setSheetsUrlState] = useState('');
  const [pipeline, setPipelineState] = useState('standard');
  const [defaultFields, setDefaultFieldsState] = useState([]);
  const [newField, setNewField] = useState('');
  const [savingState, setSavingState] = useState('idle'); // idle | saving | saved

  useEffect(() => {
    (async () => {
      const bUrl = await getBackendUrl();
      const sUrl = await getSheetsUrl();
      const pip = await getPipeline();
      const df = await getDefaultFields();
      setBackendUrlState(bUrl || '');
      setSheetsUrlState(sUrl || '');
      setPipelineState(pip);
      setDefaultFieldsState(df);
    })();
  }, []);

  const addDefaultField = () => {
    const trimmed = newField.trim();
    if (trimmed && !defaultFields.includes(trimmed)) {
      setDefaultFieldsState((prev) => [...prev, trimmed]);
      setNewField('');
    }
  };

  const removeDefaultField = (field) => {
    setDefaultFieldsState((prev) => prev.filter((f) => f !== field));
  };

  const save = async () => {
    setSavingState('saving');
    try {
      if (backendUrl.trim()) {
        await saveBackendUrl(backendUrl.trim());
        setBackendUrl(backendUrl.trim());
      }
      if (sheetsUrl.trim()) {
        await saveSheetsUrl(sheetsUrl.trim());
      }
      await savePipeline(pipeline);
      await saveDefaultFields(defaultFields);

      setSavingState('saved');
      setTimeout(() => setSavingState('idle'), 2200);
    } catch (err) {
      setSavingState('idle');
      Alert.alert('Error', err.message);
    }
  };

  return (
    <View style={styles.root}>
      <GlowBackground />
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor={colors.background} />
        <ScrollView showsVerticalScrollIndicator={false}>
          <View style={styles.headerRow}>
            <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
              <Ionicons name="arrow-back" size={20} color="#fff" />
            </TouchableOpacity>
            <Text style={styles.title}>Settings</Text>
          </View>

          <Text style={styles.fieldLabel}>Backend URL</Text>
          <TextInput
            style={styles.monoInput}
            value={backendUrl}
            onChangeText={setBackendUrlState}
            placeholder="https://filetract.onrender.com"
            placeholderTextColor={colors.textGhost}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
          />
          <Text style={styles.hint}>
            Defaults to https://filetract.onrender.com. For local testing use your machine's LAN IP instead: http://192.168.x.x:5000
          </Text>

          <Text style={[styles.fieldLabel, { marginTop: 22 }]}>Export Destination URL</Text>
          <TextInput
            style={styles.monoInput}
            value={sheetsUrl}
            onChangeText={setSheetsUrlState}
            placeholder="https://script.google.com/macros/s/..."
            placeholderTextColor={colors.textGhost}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
          />
          <Text style={styles.hint}>
            Deploy the included Google Apps Script (google_apps_script/Code.gs) and paste the Web App URL here.
          </Text>

          <Text style={[styles.fieldLabel, { marginTop: 22 }]}>Default Processing Mode</Text>
          <View style={styles.segmentRow}>
            <TouchableOpacity style={styles.segmentTouchable} onPress={() => setPipelineState('standard')}>
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
            <TouchableOpacity style={styles.segmentTouchable} onPress={() => setPipelineState('patent')}>
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

          <Text style={[styles.fieldLabel, { marginTop: 22 }]}>Default Fields</Text>
          <View style={styles.fieldsContainer}>
            {defaultFields.map((field) => (
              <View key={field} style={styles.fieldChip}>
                <Text style={styles.fieldChipText}>{field}</Text>
                <TouchableOpacity onPress={() => removeDefaultField(field)}>
                  <Ionicons name="close" size={13} color="rgba(255,255,255,0.6)" />
                </TouchableOpacity>
              </View>
            ))}
          </View>
          <View style={styles.addFieldRow}>
            <TextInput
              style={styles.fieldInput}
              placeholder="Add a default field…"
              placeholderTextColor={colors.textGhost}
              value={newField}
              onChangeText={setNewField}
              onSubmitEditing={addDefaultField}
              returnKeyType="done"
            />
            <TouchableOpacity onPress={addDefaultField}>
              <LinearGradient colors={gradients.primary} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.addBtn}>
                <Ionicons name="add" size={20} color="#fff" />
              </LinearGradient>
            </TouchableOpacity>
          </View>

          <Text style={[styles.fieldLabel, { marginTop: 26 }]}>Google Sheets Setup Guide</Text>
          <View style={styles.card}>
            {SETUP_STEPS.map((step, i) => (
              <View key={i} style={styles.stepRow}>
                <View style={styles.stepNum}>
                  <Text style={styles.stepNumText}>{i + 1}</Text>
                </View>
                <Text style={styles.stepText}>{step}</Text>
              </View>
            ))}
          </View>

          <TouchableOpacity activeOpacity={0.85} onPress={save} disabled={savingState === 'saving'} style={{ marginTop: 26, marginHorizontal: 22 }}>
            <LinearGradient
              colors={savingState === 'saved' ? gradients.success : gradients.primary}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.saveBtn}
            >
              {savingState === 'saved' && <Ionicons name="checkmark" size={17} color="#fff" />}
              <Text style={styles.saveBtnText}>
                {savingState === 'saving' ? 'Saving…' : savingState === 'saved' ? 'Saved' : 'Save Changes'}
              </Text>
            </LinearGradient>
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
  fieldLabel: {
    fontFamily: fonts.bodySemi,
    fontSize: 12.5,
    color: 'rgba(255,255,255,0.6)',
    marginTop: 22,
    marginBottom: 8,
    paddingHorizontal: 22,
  },
  monoInput: {
    fontFamily: fonts.mono,
    marginHorizontal: 22,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 13,
    color: colors.text,
    fontSize: 12.5,
  },
  hint: {
    fontFamily: fonts.body,
    marginHorizontal: 22,
    marginTop: 8,
    color: colors.textFaint,
    fontSize: 11,
    lineHeight: 17,
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
  card: {
    marginHorizontal: 22,
    backgroundColor: colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 16,
  },
  stepRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
    alignItems: 'flex-start',
  },
  stepNum: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: 'rgba(167,139,250,0.12)',
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.28)',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 1,
  },
  stepNumText: {
    fontFamily: fonts.bodyBold,
    color: colors.violet,
    fontSize: 11,
  },
  stepText: {
    fontFamily: fonts.body,
    color: colors.text,
    fontSize: 13,
    lineHeight: 20,
    flex: 1,
  },
  saveBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 9,
    paddingVertical: 16,
    borderRadius: 16,
  },
  saveBtnText: {
    fontFamily: fonts.bodySemi,
    color: '#fff',
    fontSize: 15.5,
  },
});
