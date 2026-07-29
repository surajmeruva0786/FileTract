import React from 'react';
import { View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { useFonts } from 'expo-font';

import HomeScreen from './screens/HomeScreen';
import FieldsScreen from './screens/FieldsScreen';
import ProcessingScreen from './screens/ProcessingScreen';
import PreviewScreen from './screens/PreviewScreen';
import SettingsScreen from './screens/SettingsScreen';
import { colors, fontAssets } from './theme';

const Stack = createStackNavigator();

export default function App() {
  const [fontsLoaded] = useFonts(fontAssets);

  if (!fontsLoaded) {
    return <View style={{ flex: 1, backgroundColor: colors.background }} />;
  }

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="Home"
          screenOptions={{
            headerShown: false,
            cardStyle: { backgroundColor: colors.background },
            animationEnabled: true,
          }}
        >
          <Stack.Screen name="Home" component={HomeScreen} />
          <Stack.Screen name="Fields" component={FieldsScreen} />
          <Stack.Screen
            name="Processing"
            component={ProcessingScreen}
            options={{ gestureEnabled: false }}
          />
          <Stack.Screen name="Preview" component={PreviewScreen} />
          <Stack.Screen name="Settings" component={SettingsScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </GestureHandlerRootView>
  );
}
