// Global Dark Mode Management Script

function toggleDarkMode() {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('darkMode', isDark ? 'true' : 'false');
    updateDarkModeButton();
}

function updateDarkModeButton() {
    const buttons = document.querySelectorAll('#darkModeToggle');
    const isDark = document.documentElement.classList.contains('dark');
    buttons.forEach(button => {
        const moonIcon = button.querySelector('[data-icon="moon"]');
        const sunIcon = button.querySelector('[data-icon="sun"]');

        if (moonIcon && sunIcon) {
            if (isDark) {
                moonIcon.classList.add('hidden');
                sunIcon.classList.remove('hidden');
            } else {
                sunIcon.classList.add('hidden');
                moonIcon.classList.remove('hidden');
            }
            button.title = isDark ? 'Switch to light mode' : 'Switch to dark mode';
        } else {
            // Fallback for pages that haven't been updated to dual-icon markup
            button.innerHTML = isDark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
            button.title = isDark ? 'Switch to light mode' : 'Switch to dark mode';
        }
    });
}

// Initialize button state when DOM is ready
function initDarkModeUI() {
    // With dual-icon markup and early .dark application in <head>,
    // the correct icon is visible immediately; this just ensures
    // titles and any late changes are correct without changing opacity.
    updateDarkModeButton();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDarkModeUI);
} else {
    initDarkModeUI();
}

// Listen for storage changes from other tabs
window.addEventListener('storage', function(e) {
    if (e.key === 'darkMode') {
        if (localStorage.getItem('darkMode') === 'true') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
        updateDarkModeButton();
    }
});
