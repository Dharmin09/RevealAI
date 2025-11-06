import { initializeApp, getApps } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-app.js";
import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-auth.js";

const firebaseConfig = {
    apiKey: "AIzaSyCHeT09KIOi3H6FGMmX0lnLTyMBxPQ0mQk",
    authDomain: "deepfake-backend-6554f.firebaseapp.com",
    projectId: "deepfake-backend-6554f",
    storageBucket: "deepfake-backend-6554f.firebasestorage.app",
    messagingSenderId: "804069420237",
    appId: "1:804069420237:web:8c20b51a9a4f4230624d84"
};

let app;
if (!getApps().length) {
    app = initializeApp(firebaseConfig);
} else {
    app = getApps()[0];
}
const auth = getAuth(app);

// Global state for profile button
let isUserLoggedIn = false;
const AUTH_CACHE_KEY = 'revealAI_isAuthed';

function applyAuthUIVisible(isAuthed) {
    const signInEl = document.getElementById('nav-auth-signin');
    const profileEl = document.getElementById('nav-auth-profile');
    if (signInEl) signInEl.style.display = isAuthed ? 'none' : 'inline-flex';
    if (profileEl) profileEl.style.display = isAuthed ? 'inline-flex' : 'none';
    if (isAuthed) {
        loadPreviewHistory();
    } else {
        const previewDiv = document.getElementById('profile-hover-preview');
        if (previewDiv) previewDiv.classList.add('hidden');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const signInEl = document.getElementById('nav-auth-signin');
    const profileEl = document.getElementById('nav-auth-profile');
    if (!signInEl && !profileEl) return;

    // 1) Optimistic render from cache for instant UI (no flash)
    try {
        const cached = localStorage.getItem(AUTH_CACHE_KEY);
        if (cached !== null) {
            isUserLoggedIn = cached === 'true';
            applyAuthUIVisible(isUserLoggedIn);
        } else {
            // Default to signed-out if nothing cached
            applyAuthUIVisible(false);
        }
    } catch {}

    // 2) Reconcile with Firebase auth state (authoritative)
    onAuthStateChanged(auth, (user) => {
        isUserLoggedIn = !!user;
        try { localStorage.setItem(AUTH_CACHE_KEY, isUserLoggedIn ? 'true' : 'false'); } catch {}
        applyAuthUIVisible(isUserLoggedIn);
    });

    // Setup click handler for profile to show preview
    if (profileEl) {
        profileEl.addEventListener('click', (e) => {
            if (isUserLoggedIn) {
                e.preventDefault();
                e.stopPropagation();
                toggleProfilePreview();
            }
        });
    }
});

// Toggle profile preview on click
window.toggleProfilePreview = function() {
    const navAuth = document.getElementById('nav-auth-profile');
    const previewDiv = document.getElementById('profile-hover-preview');
    
    // If user is not logged in, redirect to register
    if (!isUserLoggedIn || navAuth.textContent === 'Sign Up') {
        window.location.href = 'register.html';
        return;
    }
    
    // If preview exists and user is logged in, toggle it
    if (previewDiv) {
        const isHidden = previewDiv.classList.contains('hidden');
        if (isHidden) {
            loadPreviewHistory();
            previewDiv.classList.remove('hidden');
            // Show logout button
            const logoutBtn = document.getElementById('preview-logout-btn');
            if (logoutBtn) logoutBtn.classList.remove('hidden');
        } else {
            previewDiv.classList.add('hidden');
            // Hide logout button
            const logoutBtn = document.getElementById('preview-logout-btn');
            if (logoutBtn) logoutBtn.classList.add('hidden');
        }
    }
};

// Close preview when clicking outside
document.addEventListener('click', (event) => {
    const previewDiv = document.getElementById('profile-hover-preview');
    const navAuth = document.getElementById('nav-auth-profile');
    
    if (previewDiv && navAuth && !previewDiv.classList.contains('hidden')) {
        const isClickInsidePreview = previewDiv.contains(event.target);
        const isClickOnButton = navAuth.contains(event.target);
        
        if (!isClickInsidePreview && !isClickOnButton) {
            previewDiv.classList.add('hidden');
        }
    }
});

// Navigate to profile when clicking "View Full History" link
document.addEventListener('click', (event) => {
    if (event.target.href && event.target.href.includes('profile')) {
        event.preventDefault();
        window.location.href = '/profile';
    }
});

// Load and display recent history in the preview
function loadPreviewHistory() {
    const previewHistoryDiv = document.getElementById('preview-history');
    if (!previewHistoryDiv) return;
    
    try {
        const stored = localStorage.getItem('revealAI_analysisHistory');
        
        if (!stored || stored === '[]') {
            previewHistoryDiv.innerHTML = '<p class="text-gray-500">No analysis history yet</p>';
            return;
        }
        
        const analysisHistory = JSON.parse(stored);
        
        if (analysisHistory.length === 0) {
            previewHistoryDiv.innerHTML = '<p class="text-gray-500">No analysis history yet</p>';
            return;
        }
        
        // Show last 5 items
        const recentItems = analysisHistory.slice(-5).reverse();
        const previewHTML = recentItems.map(item => {
            const score = item.score ? Math.round(item.score * 100) : 0;
            return `
                <div class="border-l-4 border-indigo-600 pl-3 py-2">
                    <p class="font-semibold text-gray-800 truncate text-sm">${item.filename || 'Unknown'}</p>
                    <p class="text-xs text-gray-500">${item.type || 'unknown'} • ${score}%</p>
                    <p class="text-xs font-medium" style="color: ${item.result && item.result.toLowerCase().includes('deepfake') ? '#ef4444' : '#f59e0b'}">${item.result || 'Unknown'}</p>
                </div>
            `;
        }).join('');
        
        previewHistoryDiv.innerHTML = previewHTML;
    } catch (e) {
        console.error('Error loading preview history:', e);
        previewHistoryDiv.innerHTML = '<p class="text-gray-500">No analysis history yet</p>';
    }
}

// Logout function
window.logoutUser = async function() {
    try {
        const auth = getAuth();
        await signOut(auth);
        // Clear local storage related to user session
        localStorage.removeItem('revealAI_userToken');
        localStorage.removeItem('revealAI_analysisHistory');
        try { localStorage.setItem(AUTH_CACHE_KEY, 'false'); } catch {}
        console.log('User logged out successfully.');
        window.location.href = '/'; // Redirect to home page
    } catch (error) {
        console.error('Error logging out:', error);
        alert('Error logging out. Please try again.');
    }
};
