// API Base URL (Render backend)
const API_BASE = 'https://insect-identifier-api.onrender.com';

// Select DOM elements
const video = document.getElementById('camera-feed');
const canvas = document.getElementById('capture-canvas');
const uploadImg = document.getElementById('uploaded-image');
const viewGuide = document.getElementById('view-guide');

// Buttons
const takePhotoBtn = document.getElementById('take-photo-btn');
const retakePhotoBtn = document.getElementById('retake-photo-btn');
const uploadBtn = document.getElementById('upload-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const fileInput = document.getElementById('file-input');
const loadingSpinner = document.getElementById('loading-spinner');

// Result fields
const statusDisplay = document.getElementById('status-display');
const confidenceDisplay = document.getElementById('confidence-display');
const identifiedInsect = document.getElementById('identified-insect');
const suggestedTreatment = document.getElementById('suggested-treatment');

let stream = null;
let currentImageDataUrl = null;

// Initialize Camera
async function initCamera() {
    try {
        if (stream) stream.getTracks().forEach(track => track.stop());

        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
        });

        video.srcObject = stream;
        video.style.display = 'block';
        uploadImg.style.display = 'none';
        viewGuide.style.display = 'block';
        
        takePhotoBtn.style.display = 'flex';
        retakePhotoBtn.style.display = 'none';
        currentImageDataUrl = null;
        updateAnalyzeBtn();

    } catch (err) {
        console.error("Camera error:", err);
    }
}

// Function to enable/disable analyze button
function updateAnalyzeBtn() {
    if (currentImageDataUrl) {
        analyzeBtn.style.opacity = '1';
        analyzeBtn.style.pointerEvents = 'auto';
        analyzeBtn.innerHTML = "✨ Analyze Image ✨";
    } else {
        analyzeBtn.style.opacity = '0.5';
        analyzeBtn.style.pointerEvents = 'none';
    }
}

// Start Stream on load
initCamera();

// Capture Photo Event
takePhotoBtn.addEventListener('click', () => {
    const size = Math.min(video.videoWidth, video.videoHeight);
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    // Crop center
    const startX = (video.videoWidth - size) / 2;
    const startY = (video.videoHeight - size) / 2;

    ctx.drawImage(video, startX, startY, size, size, 0, 0, size, size);
    
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    setReadyImage(dataUrl);
});

// File Upload Event
uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            // Resize image via canvas before sending to save massive bandwidth and memory!
            const img = new Image();
            img.onload = () => {
                const MAX_SIZE = 600; // Perfect for 224x224 neural net input
                let width = img.width;
                let height = img.height;

                if (width > height && width > MAX_SIZE) {
                    height *= MAX_SIZE / width;
                    width = MAX_SIZE;
                } else if (height > MAX_SIZE) {
                    width *= MAX_SIZE / height;
                    height = MAX_SIZE;
                }

                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);
                // Compress output slightly to save backend JSON parsing memory
                const compressedDataUrl = canvas.toDataURL('image/jpeg', 0.85);
                setReadyImage(compressedDataUrl);
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// Set active image
function setReadyImage(dataUrl) {
    currentImageDataUrl = dataUrl;
    uploadImg.src = dataUrl;
    
    video.style.display = 'none';
    viewGuide.style.display = 'none';
    uploadImg.style.display = 'block';
    
    takePhotoBtn.style.display = 'none';
    retakePhotoBtn.style.display = 'flex';
    
    // Stop camera to save battery
    if (stream) stream.getTracks().forEach(track => track.stop());
    
    updateAnalyzeBtn();
}

// Retake / Reset 
retakePhotoBtn.addEventListener('click', () => {
    fileInput.value = ''; // Reset file input
    initCamera();
});

// Analyze Click
analyzeBtn.addEventListener('click', async () => {
    if (!currentImageDataUrl) return;
    
    loadingSpinner.style.display = 'flex';
    analyzeBtn.style.opacity = '0.5';
    analyzeBtn.style.pointerEvents = 'none';
    analyzeBtn.innerHTML = "Processing...";
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: currentImageDataUrl })
        });

        const result = await response.json();
        
        if (result.success) {
            const insectName = result.prediction.replace(/_/g, ' ');
            
            statusDisplay.textContent = "Analyzed successfully";
            statusDisplay.classList.add('active');
            statusDisplay.style.color = "var(--accent-green)";
            
            confidenceDisplay.textContent = `Confidence: ${result.confidence.toFixed(1)}%`;
            identifiedInsect.textContent = insectName.toUpperCase();
            
            if (result.details && result.details !== "Details not found for this species.") {
                suggestedTreatment.innerHTML = `<div class="details-content">${result.details}</div>`;
            } else {
                suggestedTreatment.innerHTML = `Identified as <strong>${insectName}</strong>.<br>No further details found in the database.`;
            }
        } else {
            statusDisplay.textContent = "Error";
            statusDisplay.style.color = "#ef4444";
            identifiedInsect.textContent = "Server Error";
            suggestedTreatment.textContent = result.error;
        }

    } catch (error) {
        statusDisplay.textContent = "Network Error";
        statusDisplay.style.color = "#ef4444";
        identifiedInsect.textContent = "Offline";
        suggestedTreatment.textContent = "Could not reach the analysis server. Please ensure the server is running.";
    } finally {
        loadingSpinner.style.display = 'none';
        updateAnalyzeBtn();
    }
});
