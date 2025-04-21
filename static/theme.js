// Theme switching functionality
document.addEventListener('DOMContentLoaded', function() {
    // Check for saved theme preference or use default
    const currentTheme = localStorage.getItem('theme') || 'light';

    // Apply the theme
    if (currentTheme === 'dark') {
        document.body.classList.add('dark-mode');
        document.body.classList.remove('light-mode');
    } else {
        document.body.classList.add('light-mode');
        document.body.classList.remove('dark-mode');
    }

    // Add event listeners to all theme toggle buttons
    const themeToggles = document.querySelectorAll('.theme-toggle');
    themeToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            // Toggle the theme
            if (document.body.classList.contains('light-mode')) {
                document.body.classList.remove('light-mode');
                document.body.classList.add('dark-mode');
                localStorage.setItem('theme', 'dark');
                updateToggleIcons('dark');
            } else {
                document.body.classList.remove('dark-mode');
                document.body.classList.add('light-mode');
                localStorage.setItem('theme', 'light');
                updateToggleIcons('light');
            }
        });
    });

    // Update toggle button icons based on current theme
    updateToggleIcons(currentTheme);
});

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
