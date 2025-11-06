// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-app.js";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries
import { getAuth, createUserWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-auth.js";

// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyCHeT09KIOi3H6FGMmX0lnLTyMBxPQ0mQk",
    authDomain: "deepfake-backend-6554f.firebaseapp.com",
    projectId: "deepfake-backend-6554f",
    storageBucket: "deepfake-backend-6554f.firebasestorage.app",
    messagingSenderId: "804069420237",
    appId: "1:804069420237:web:8c20b51a9a4f4230624d84"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);



// Form submit event (handles button click and Enter key)
const form = document.querySelector('.auth-form');
form.addEventListener('submit', function(event) {
    event.preventDefault();
    const email = document.getElementById('reg-email').value;
    const password = document.getElementById('reg-password').value;
    createUserWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
            // Signed up 
            const user = userCredential.user;
            alert("Creating Account...");
            window.location.href = "index.html";
            // ...
        })
        .catch((error) => {
            const errorMessage = error.message;
            alert(errorMessage);
            // ..
        });
});
