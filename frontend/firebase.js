// ═══════════════════════════════════════════════════════
//  CineMate — Firebase Auth Module
//  ⚠️  Fill in your Firebase project config below.
//  Get it from: Firebase Console → Project Settings → Your Apps → SDK setup
// ═══════════════════════════════════════════════════════

import { initializeApp } from 'firebase/app';
import {
    getAuth,
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    sendPasswordResetEmail,
    updateProfile,
    signOut,
    onAuthStateChanged,
    GoogleAuthProvider,
    signInWithPopup,
} from 'firebase/auth';

// ── YOUR FIREBASE CONFIG ────────────────────────────────
// Replace every value below with your own project config
const firebaseConfig = {
    apiKey: "AIzaSyAh9HlbqNCcXchQ-Eplmalzw5qsC7ZOkVs",
    authDomain: "cinemate-45c26.firebaseapp.com",
    projectId: "cinemate-45c26",
    storageBucket: "cinemate-45c26.firebasestorage.app",
    messagingSenderId: "973910089721",
    appId: "1:973910089721:web:965d53027a270c773ba728",
    measurementId: "G-V2LZ5EF697",
};
// ── END CONFIG ─────────────────────────────────────────

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();

// ── AUTH HELPERS ────────────────────────────────────────

/** Register with email + password. Sets display name to username. */
export async function firebaseRegister(email, password, username) {
    const cred = await createUserWithEmailAndPassword(auth, email, password);
    await updateProfile(cred.user, { displayName: username });
    return cred.user;
}

/** Login with email + password. */
export async function firebaseLogin(email, password) {
    const cred = await signInWithEmailAndPassword(auth, email, password);
    return cred.user;
}

/** Sign in with Google popup. */
export async function firebaseLoginGoogle() {
    const cred = await signInWithPopup(auth, googleProvider);
    return cred.user;
}

/** Send a password reset email. */
export async function firebaseForgotPassword(email) {
    await sendPasswordResetEmail(auth, email);
}

/** Sign out. */
export async function firebaseLogout() {
    await signOut(auth);
}

/** Get the current user's Firebase ID token (for backend verification). */
export async function getIdToken() {
    const u = auth.currentUser;
    if (!u) return null;
    return u.getIdToken(/* forceRefresh= */ false);
}

/** Listen to auth state changes. */
export function onAuthChange(callback) {
    return onAuthStateChanged(auth, callback);
}
