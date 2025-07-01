# PowerShell script to extract YouTube cookies using yt-dlp
# Run this on your Windows machine where you're logged into YouTube

Write-Host "Extracting YouTube cookies..." -ForegroundColor Green

# Check if yt-dlp is installed
if (!(Get-Command yt-dlp -ErrorAction SilentlyContinue)) {
    Write-Host "yt-dlp not found. Please install it first:" -ForegroundColor Red
    Write-Host "pip install yt-dlp" -ForegroundColor Yellow
    exit 1
}

# Extract cookies from Chrome (change to 'edge' if using Edge)
$browser = "chrome"
$outputFile = "youtube_cookies.txt"

Write-Host "Extracting cookies from $browser..." -ForegroundColor Yellow

# Run yt-dlp to extract cookies
yt-dlp --cookies-from-browser $browser --cookies $outputFile --skip-download "https://www.youtube.com"

if (Test-Path $outputFile) {
    Write-Host "Cookies extracted successfully to: $outputFile" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Copy this file to your server:"
    Write-Host "   scp $outputFile root@167.99.1.8:~/Mnemo/data/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "2. Or copy the contents and paste on server:"
    Write-Host "   cat $outputFile" -ForegroundColor Yellow
} else {
    Write-Host "Failed to extract cookies" -ForegroundColor Red
}