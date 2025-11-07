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

const app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
const auth = getAuth(app);

const profileInfo = document.getElementById('profile-info');
const logoutBtn = document.getElementById('logout-btn');

function renderLoggedOutState() {
    if (profileInfo) {
        profileInfo.innerHTML = '<p>Please log in to see your profile information.</p>';
    }
    if (logoutBtn) {
        logoutBtn.style.display = 'none';
    }
}

function renderUserProfile(user) {
    if (!profileInfo) {
        return;
    }

    const created = user.metadata?.creationTime || 'Unknown';
    const lastLogin = user.metadata?.lastSignInTime || 'Unknown';

    profileInfo.innerHTML = `
        <p><strong>Email:</strong> ${user.email}</p>
        <p><strong>UID:</strong> ${user.uid}</p>
        <p><strong>Created:</strong> ${created}</p>
        <p><strong>Last Login:</strong> ${lastLogin}</p>
    `;

    if (logoutBtn) {
        logoutBtn.style.display = 'inline-flex';
        logoutBtn.onclick = async () => {
            try {
                await signOut(auth);
                renderLoggedOutState();
                window.location.href = '/';
            } catch (err) {
                console.error('Error during logout', err);
                alert('Failed to log out. Please try again.');
            }
        };
    }
}

onAuthStateChanged(auth, (user) => {
    if (!user) {
        renderLoggedOutState();
        return;
    }

    renderUserProfile(user);
});
