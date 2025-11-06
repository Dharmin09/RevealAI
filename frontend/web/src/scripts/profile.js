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

function renderLearningStatus(status) {
    if (!profileInfo) {
        return;
    }

    let statusContainer = document.getElementById('learning-status');
    if (!statusContainer) {
        statusContainer = document.createElement('div');
        statusContainer.id = 'learning-status';
        statusContainer.className = 'mt-6 p-4 rounded-lg bg-indigo-50 border border-indigo-200 text-sm text-gray-700';
        profileInfo.appendChild(statusContainer);
    }

    const perf = status?.current_performance || {};
    const improvement = status?.improvement_metrics || {};
    const retrain = status?.retraining || {};

    const accuracy = perf.accuracy !== undefined ? `${perf.accuracy}%` : 'N/A';
    const totalLogged = perf.total ?? 0;
    const correct = perf.correct ?? 0;
    const incorrect = perf.incorrect ?? 0;

    const improvementCopy = improvement?.insufficient_data
        ? `Need ${improvement.predictions_needed} more labeled predictions for trend analysis.`
        : `Accuracy improved by ${improvement.improvement ?? 0}% (recent period)`;

    const autoRetrainCopy = retrain?.status === 'running'
        ? `Retraining models… started ${retrain.started_at || 'recently'}.`
        : retrain?.status === 'ready'
            ? 'Corrections ready. Retraining queued.'
            : retrain?.status === 'idle'
                ? 'Retraining not required right now.'
                : 'Retraining status unknown.';

    statusContainer.innerHTML = `
        <h3 class="text-base font-semibold text-indigo-700 mb-2">Model Learning Health</h3>
        <p><strong>Recent Accuracy:</strong> ${accuracy}</p>
        <p><strong>Predictions Logged:</strong> ${totalLogged} (✅ ${correct} / ❌ ${incorrect})</p>
        <p><strong>Trend:</strong> ${improvementCopy}</p>
        <p><strong>Auto-Retrain:</strong> ${autoRetrainCopy}</p>
    `;
}

async function fetchLearningStatus() {
    try {
        const response = await fetch('/api/learning/status', { cache: 'no-store' });
        if (!response.ok) {
            throw new Error(`Status request failed: ${response.status}`);
        }
        const data = await response.json();
        renderLearningStatus(data);
    } catch (err) {
        console.warn('Unable to fetch learning status', err);
    }
}

onAuthStateChanged(auth, (user) => {
    if (!user) {
        renderLoggedOutState();
        return;
    }

    renderUserProfile(user);
    fetchLearningStatus();
});

// Fallback: if auth state takes too long, still attempt to load status so admins can monitor
setTimeout(fetchLearningStatus, 3000);
