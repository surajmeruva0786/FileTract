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
  Switch,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
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

export default function SettingsScreen({ navigation }) {
  const [backendUrl, setBackendUrlState] = useState('');
  const [sheetsUrl, setSheetsUrlState] = useState('');
  const [pipeline, setPipelineState] = useState('standard');
  const [defaultFields, setDefaultFieldsState] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    (async () => {
      const bUrl = await getBackendUrl();
      const sUrl = await getSheetsUrl();
      const pip = await getPipeline();
      const df = await getDefaultFields();
      setBackendUrlState(bUrl || '');
      setSheetsUrlState(sUrl || '');
      setPipelineState(pip);
      setDefaultFieldsState(df.join(', '));
    })();
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      if (backendUrl.trim()) {
        await saveBackendUrl(backendUrl.trim());
        setBackendUrl(backendUrl.trim());
      }
      if (sheetsUrl.trim()) {
        await saveSheetsUrl(sheetsUrl.trim());
      }
      await savePipeline(pipeline);
      const fieldsArr = defaultFields
        .split(',')
        .map((f) => f.trim())
        .filter(Boolean);
      await saveDefaultFields(fieldsArr);

      Alert.alert('Saved', 'Settings have been saved successfully.');
    } catch (err) {
      Alert.alert('Error', err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={colors.background} />
      <ScrollView showsVerticalScrollIndicator={false}>
        <View style={styles.headerRow}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
            <Ionicons name="arrow-back" size={24} color={colors.accent} />
          </TouchableOpacity>
          <Text style={styles.title}>Settings</Text>
        </View>

        {/* Backend Config */}
        <Text style={styles.sectionLabel}>FileTract Backend</Text>
        <View style={styles.card}>
          <Text style={styles.cardLabel}>Backend URL</Text>
          <TextInput
            style={styles.input}
            value={backendUrl}
            onChangeText={setBackendUrlState}
            placeholder="https://your-app.onrender.com"
            placeholderTextColor={colors.textDim}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
          />
          <Text style={styles.hint}>
            Point this to your FileTract Flask backend.{'\n'}
            For local testing use your machine's LAN IP: http://192.168.x.x:5000
          </Text>
        </View>

        {/* Google Sheets Config */}
        <Text style={styles.sectionLabel}>Google Sheets Integration</Text>
        <View style={styles.card}>
          <Text style={styles.cardLabel}>Apps Script Web App URL</Text>
          <TextInput
            style={styles.input}
            value={sheetsUrl}
            onChangeText={setSheetsUrlState}
            placeholder="https://script.google.com/macros/s/..."
            placeholderTextColor={colors.textDim}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
          />
          <Text style={styles.hint}>
            Deploy the included Google Apps Script (google_apps_script/Code.gs) and paste the Web App URL here.
          </Text>
        </View>

        {/* Pipeline Default */}
        <Text style={styles.sectionLabel}>Default Pipeline</Text>
        <View style={styles.card}>
          <View style={styles.pipelineRow}>
            <TouchableOpacity
              style={[styles.pipelineBtn, pipeline === 'standard' && styles.pipelineBtnActive]}
              onPress={() => setPipelineState('standard')}
            >
              <Ionicons
                name="flash"
                size={18}
                color={pipeline === 'standard' ? colors.background : colors.textDim}
              />
              <Text
                style={[styles.pipelineBtnText, pipeline === 'standard' && styles.pipelineBtnTextActive]}
              >
                Standard (Fast)
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.pipelineBtn, pipeline === 'patent' && styles.pipelineBtnActive]}
              onPress={() => setPipelineState('patent')}
            >
              <Ionicons
                name="shield-checkmark"
                size={18}
                color={pipeline === 'patent' ? colors.background : colors.accent}
              />
              <Text
                style={[styles.pipelineBtnText, pipeline === 'patent' && styles.pipelineBtnTextActive]}
              >
                Patent (Accurate)
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Default Fields */}
        <Text style={styles.sectionLabel}>Default Fields</Text>
        <View style={styles.card}>
          <Text style={styles.cardLabel}>Pre-filled field list (comma-separated)</Text>
          <TextInput
            style={[styles.input, { minHeight: 80 }]}
            value={defaultFields}
            onChangeText={setDefaultFieldsState}
            placeholder="Name, ID Number, Date of Birth, Expiry Date"
            placeholderTextColor={colors.textDim}
            multiline
          />
        </View>

        {/* How to set up Google Sheets */}
        <Text style={styles.sectionLabel}>Google Sheets Setup Guide</Text>
        <View style={styles.card}>
          {[
            'Open your Google Sheet.',
            'Go to Extensions → Apps Script.',
            'Paste the contents of filetract_mobile/google_apps_script/Code.gs.',
            'Click Deploy → New Deployment.',
            'Select type: Web App.',
            'Set "Execute as": Me, "Who has access": Anyone.',
            'Click Deploy and copy the Web App URL.',
            'Paste it in the field above and save.',
          ].map((step, i) => (
            <View key={i} style={styles.stepRow}>
              <View style={styles.stepNum}>
                <Text style={styles.stepNumText}>{i + 1}</Text>
              </View>
              <Text style={styles.stepText}>{step}</Text>
            </View>
          ))}
        </View>

        <TouchableOpacity style={styles.saveBtn} onPress={save} disabled={saving}>
          <Text style={styles.saveBtnText}>{saving ? 'Saving...' : 'Save Settings'}</Text>
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
    fontSize: 22,
    fontWeight: '800',
    color: colors.text,
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
  card: {
    marginHorizontal: 20,
    backgroundColor: colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 16,
  },
  cardLabel: {
    color: colors.textDim,
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 8,
  },
  input: {
    backgroundColor: colors.background,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    color: colors.text,
    paddingHorizontal: 14,
    paddingVertical: 11,
    fontSize: 13,
    textAlignVertical: 'top',
  },
  hint: {
    color: colors.textDim,
    fontSize: 11,
    marginTop: 8,
    lineHeight: 17,
  },
  pipelineRow: {
    flexDirection: 'row',
    gap: 10,
  },
  pipelineBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 12,
    borderRadius: 10,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
  },
  pipelineBtnActive: {
    backgroundColor: colors.accent,
    borderColor: colors.accent,
  },
  pipelineBtnText: {
    color: colors.textDim,
    fontSize: 12,
    fontWeight: '700',
  },
  pipelineBtnTextActive: {
    color: colors.background,
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
    backgroundColor: colors.accentGlow,
    borderWidth: 1,
    borderColor: colors.accent + '44',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 1,
  },
  stepNumText: {
    color: colors.accent,
    fontSize: 11,
    fontWeight: '800',
  },
  stepText: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 20,
    flex: 1,
  },
  saveBtn: {
    backgroundColor: colors.accent,
    marginHorizontal: 20,
    marginTop: 28,
    paddingVertical: 18,
    borderRadius: 14,
    alignItems: 'center',
  },
  saveBtnText: {
    color: colors.background,
    fontSize: 16,
    fontWeight: '800',
  },
});
