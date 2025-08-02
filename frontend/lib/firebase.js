// Firebase configuration for the web app
import { initializeApp } from "firebase/app";
import { getStorage, ref, uploadBytes, listAll, getDownloadURL } from "firebase/storage";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBwQVi2V_UWbRLDZo9Kqp_8Q9ptswuYp5s",
  authDomain: "notechat-26c38.firebaseapp.com",
  projectId: "notechat-26c38",
  storageBucket: "notechat-26c38.firebasestorage.app",
  messagingSenderId: "1019455193571",
  appId: "1:1019455193571:web:3ca7e557abd32848d7e0d6",
  measurementId: "G-1J7NVX9XZQ"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const storage = getStorage(app);

export { storage, ref, uploadBytes, listAll, getDownloadURL };
