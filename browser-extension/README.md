# Mnemo Cookie Sync Extension

This Chrome/Edge extension automatically syncs your YouTube cookies to your Mnemo server, so you never have to manually deal with cookies again!

## Installation

1. Open Chrome/Edge and go to `chrome://extensions/` (or `edge://extensions/`)
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `browser-extension` folder

## How it works

1. **Automatic Sync**: When you browse YouTube, cookies are automatically sent to your Mnemo server
2. **Manual Sync**: Click the extension icon and hit "Sync YouTube Cookies"
3. **Smart Updates**: Only syncs when cookies change (max once per minute)

## Configuration

To change the server URL:
1. Edit `background.js`
2. Change `MNEMO_SERVER` to your server's URL
3. Reload the extension

## Privacy

- Only YouTube cookies are accessed
- Cookies are sent only to YOUR Mnemo server
- No third-party access

## Icons

For now, the extension works without icons. To add icons:
1. Create 16x16, 48x48, and 128x128 PNG images
2. Name them icon16.png, icon48.png, icon128.png
3. Place in the extension folder