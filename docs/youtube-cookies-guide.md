# YouTube Cookie Authentication Guide

## Quick Start

### Option 1: Browser Extension (Easiest)
1. Install "Get cookies.txt LOCALLY" extension for Chrome/Edge
2. Go to YouTube and sign in
3. Click the extension icon
4. Click "Export" or download cookies
5. Save as `youtube_cookies.txt`

### Option 2: Manual Export from Chrome/Edge
1. Go to YouTube and sign in
2. Press F12 to open Developer Tools
3. Go to Application tab → Cookies → https://www.youtube.com
4. You need these cookies: `VISITOR_INFO1_LIVE`, `PREF`, `LOGIN_INFO`, `SID`, `HSID`, `SSID`, `APISID`, `SAPISID`

### Option 3: Using yt-dlp directly
```bash
# On your Windows machine, extract cookies:
yt-dlp --cookies-from-browser chrome --cookies cookies.txt --skip-download "https://www.youtube.com"
```

## Deploying Cookies to Server

### Method 1: Direct Upload
```bash
# From your local machine
scp youtube_cookies.txt root@167.99.1.8:~/Mnemo/data/
```

### Method 2: Copy-Paste via Console
```bash
# On server, create the file
cat > ~/Mnemo/data/youtube_cookies.txt << 'EOF'
# Paste your cookies here in Netscape format
EOF
```

### Method 3: Via Web Interface (Future)
We'll add an API endpoint to upload cookies securely.

## Cookie Format (Netscape)
```
# Netscape HTTP Cookie File
.youtube.com	TRUE	/	TRUE	1234567890	COOKIE_NAME	cookie_value
```

## Security Notes
- Cookies expire after ~2 weeks
- Don't commit cookies to git
- Keep cookies file permissions restricted: `chmod 600 youtube_cookies.txt`
- Consider rotating cookies regularly

## Testing
After adding cookies, test with a protected video:
```bash
curl -X POST http://localhost/api/v1/video/process \
  -H "Content-Type: application/json" \
  -d '{"video_url": "YOUR_VIDEO_URL", "processing_level": "basic"}'
```