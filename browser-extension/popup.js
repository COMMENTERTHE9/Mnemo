// Popup script for Mnemo Cookie Sync

document.getElementById('syncButton').addEventListener('click', () => {
  const statusDiv = document.getElementById('status');
  const button = document.getElementById('syncButton');
  
  // Disable button and show loading
  button.disabled = true;
  statusDiv.className = 'info';
  statusDiv.textContent = 'Syncing cookies...';
  
  // Send message to background script
  chrome.runtime.sendMessage({ action: 'syncNow' }, (response) => {
    button.disabled = false;
    
    if (response && response.success) {
      statusDiv.className = 'success';
      statusDiv.textContent = 'Cookies synced successfully!';
    } else {
      statusDiv.className = 'error';
      statusDiv.textContent = 'Failed to sync cookies';
    }
    
    // Clear status after 3 seconds
    setTimeout(() => {
      statusDiv.textContent = '';
      statusDiv.className = '';
    }, 3000);
  });
});

// Check last sync time on popup open
chrome.storage.local.get(['lastSync'], (result) => {
  if (result.lastSync) {
    const lastSyncDate = new Date(result.lastSync);
    const statusDiv = document.getElementById('status');
    statusDiv.className = 'info';
    statusDiv.textContent = `Last sync: ${lastSyncDate.toLocaleTimeString()}`;
  }
});