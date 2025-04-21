// JavaScript function to clear logs directly
function clearLogsDirectly() {
    if (confirm('Are you sure you want to clear all logs? This action cannot be undone.')) {
        // Show loading indicator
        document.getElementById('clearStatus').innerHTML = '<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Clearing logs...</div>';
        
        // Make a POST request to the clear logs action
        fetch('/clear_logs_action', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (response.redirected) {
                // If we're redirected, go to the new URL
                window.location.href = response.url;
            } else {
                // Show success message
                document.getElementById('clearStatus').innerHTML = '<div class="alert alert-success"><i class="fas fa-check-circle"></i> Logs cleared successfully!</div>';
                
                // Redirect after a short delay
                setTimeout(() => {
                    window.location.href = '/access_logs';
                }, 2000);
            }
        })
        .catch(error => {
            // Show error message
            document.getElementById('clearStatus').innerHTML = '<div class="alert alert-danger"><i class="fas fa-exclamation-circle"></i> Error clearing logs: ' + error.message + '</div>';
        });
    }
}
