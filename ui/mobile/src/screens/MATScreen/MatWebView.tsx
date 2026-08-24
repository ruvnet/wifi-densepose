import { StyleProp, ViewStyle } from 'react-native';
import WebView, { type WebViewMessageEvent } from 'react-native-webview';
import { Asset } from 'expo-asset';
import type { RefObject } from 'react';
import MAT_DASHBOARD_HTML from '@/assets/webview/mat-dashboard.html';

const matDashboardAsset = Asset.fromModule(MAT_DASHBOARD_HTML);

type MatWebViewProps = {
  webViewRef: RefObject<WebView | null>;
  onMessage: (event: WebViewMessageEvent) => void;
  style?: StyleProp<ViewStyle>;
};

export const MatWebView = ({ webViewRef, onMessage, style }: MatWebViewProps) => {
  return (
    <WebView
      ref={webViewRef}
      originWhitelist={["*"]}
      style={style}
      source={{ uri: matDashboardAsset.uri }}
      onMessage={onMessage}
      javaScriptEnabled
      domStorageEnabled
      mixedContentMode="always"
      overScrollMode="never"
    />
  );
};
