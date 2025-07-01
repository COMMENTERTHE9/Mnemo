// Options page script

// Load saved settings
chrome.storage.sync.get(['serverUrl'], (result) => {
  if (result.serverUrl) {
    document.getElementById('serverUrl').value = result.serverUrl;
  }
});

// Save settings
document.getElementById('saveButton').addEventListener('click', () => {
  const serverUrl = document.getElementById('serverUrl').value.trim();
  const statusDiv = document.getElementById('status');
  
  if (!serverUrl) {
    statusDiv.className = 'error';
    statusDiv.textContent = 'Please enter a server URL';
    statusDiv.style.display = 'block';
    return;
  }
  
  // Validate URL format
  try {
    new URL(serverUrl);
  } catch (e) {
    statusDiv.className = 'error';
    statusDiv.textContent = 'Please enter a valid URL';
    statusDiv.style.display = 'block';
    return;
  }
  
  // Save to Chrome storage
  chrome.storage.sync.set({ serverUrl }, () => {
    statusDiv.className = 'success';
    statusDiv.textContent = 'Settings saved successfully!';
    statusDiv.style.display = 'block';
    
    // Notify background script
    chrome.runtime.sendMessage({ action: 'settingsUpdated' });
    
    // Hide status after 3 seconds
    setTimeout(() => {
      statusDiv.style.display = 'none';
    }, 3000);
  });
});