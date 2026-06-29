import AsyncStorage from '@react-native-async-storage/async-storage';

const BACKEND_URL_KEY = '@filetract_backend_url';
const PIPELINE_KEY = '@filetract_pipeline';
const DEFAULT_FIELDS_KEY = '@filetract_default_fields';

export async function saveBackendUrl(url) {
  await AsyncStorage.setItem(BACKEND_URL_KEY, url);
}

export async function getBackendUrl() {
  return await AsyncStorage.getItem(BACKEND_URL_KEY);
}

export async function savePipeline(pipeline) {
  await AsyncStorage.setItem(PIPELINE_KEY, pipeline);
}

export async function getPipeline() {
  return (await AsyncStorage.getItem(PIPELINE_KEY)) || 'standard';
}

export async function saveDefaultFields(fields) {
  await AsyncStorage.setItem(DEFAULT_FIELDS_KEY, JSON.stringify(fields));
}

export async function getDefaultFields() {
  const raw = await AsyncStorage.getItem(DEFAULT_FIELDS_KEY);
  return raw ? JSON.parse(raw) : ['Name', 'ID Number', 'Date of Birth', 'Expiry Date'];
}
