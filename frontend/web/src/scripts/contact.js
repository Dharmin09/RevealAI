import { initializeApp, getApps } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-app.js";
import { getFirestore, collection, addDoc, serverTimestamp, enableNetwork } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-firestore.js";

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
const db = getFirestore(app);

// Enable network on initialization
enableNetwork(db).catch(err => console.log('Network already enabled:', err));

const contactForm = document.getElementById('contactForm');
const nameInput = document.getElementById('name');
const emailInput = document.getElementById('email');
const subjectInput = document.getElementById('subject');
const messageInput = document.getElementById('message');

if (contactForm) {
    contactForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        
        // Validate inputs
        if (!nameInput.value.trim() || !emailInput.value.trim() || !subjectInput.value.trim() || !messageInput.value.trim()) {
            alert("Please fill in all fields.");
            return;
        }
        
        try {
            console.log('Submitting contact form...');
            console.log('Name:', nameInput.value);
            console.log('Email:', emailInput.value);
            console.log('Subject:', subjectInput.value);
            console.log('Message:', messageInput.value);
            
            const docRef = await addDoc(collection(db, 'contactform'), {
                name: nameInput.value.trim(),
                email: emailInput.value.trim(),
                subject: subjectInput.value.trim(),
                message: messageInput.value.trim(),
                timestamp: serverTimestamp(),
                status: 'new'
            });
            
            console.log('Message sent successfully with ID:', docRef.id);
            contactForm.reset();
            alert("✅ Message sent successfully! We'll get back to you soon.");
            
        } catch (error) {
            console.error('Error details:', error);
            console.error('Error code:', error.code);
            console.error('Error message:', error.message);
            
            // Provide more helpful error messages
            if (error.code === 'permission-denied') {
                alert("⚠️ Permission Error: Our Firestore rules may be misconfigured. Please contact support or try again later.");
            } else if (error.code === 'unavailable') {
                alert("⚠️ Service temporarily unavailable. Please try again in a few moments.");
            } else {
                alert("❌ Failed to send message. Error: " + error.message);
            }
        }
    });
} else {
    console.error('Contact form element not found with ID: contactForm');
}