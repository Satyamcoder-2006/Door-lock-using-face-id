// Theme switching functionality
document.addEventListener('DOMContentLoaded', function() {
    // Check for saved theme preference or use default
    const currentTheme = localStorage.getItem('theme') || 'light';
    applyTheme(currentTheme);

    // Add event listeners to all theme toggle buttons
    const themeToggles = document.querySelectorAll('.theme-toggle');
    themeToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            // Toggle the theme
            const newTheme = document.body.classList.contains('dark-mode') ? 'light' : 'dark';
            applyTheme(newTheme);
            localStorage.setItem('theme', newTheme);
            
            // Also update server-side session if the toggle_theme route exists
            // This ensures both client and server stay in sync
            fetch('/toggle_theme', { method: 'GET' })
                .then(response => {
                    console.log('Theme preference updated on server');
                })
                .catch(error => console.log('Note: Server-side theme toggle not available'));
        });
    });
});

// Function to apply theme consistently
function applyTheme(theme) {
    if (theme === 'dark') {
        document.body.classList.remove('light-mode');
        document.body.classList.add('dark-mode');
    } else {
        document.body.classList.remove('dark-mode');
        document.body.classList.add('light-mode');
    }
    updateToggleIcons(theme);
    
    // Dispatch a custom event that other components can listen for
    document.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme } }));
}

// Update the icons on theme toggle buttons
function updateToggleIcons(theme) {
    const themeToggles = document.querySelectorAll('.theme-toggle');
    themeToggles.forEach(toggle => {
        if (theme === 'dark') {
            toggle.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
        } else {
            toggle.innerHTML = '<i class="fas fa-moon"></i> Dark Mode';
        }
    });
}

// Function to get current theme - can be used by other scripts
function getCurrentTheme() {
    return document.body.classList.contains('dark-mode') ? 'dark' : 'light';
}
