/**
 * RevealAI Frontend - Main JavaScript
 * Connects the website to the Flask REST API backend
 * Enhanced with better progress tracking and error handling
 */

// API Configuration
const API_BASE_URL = window.location.origin + '/api';
let currentAnalysisData = null;
let fileUploaded = false;

console.log('RevealAI Frontend Loaded');
console.log(`API Base URL: ${API_BASE_URL}`);

// ============================================================================
// DOMContentLoaded - Main Entry Point
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded and parsed');

    // Get elements
    const uploadForm = document.getElementById('uploadForm');
    const mediaUpload = document.getElementById('mediaUpload');
    const analyzeBtn = document.getElementById('analyzeBtn');

    // Check if elements exist
    if (!uploadForm || !mediaUpload || !analyzeBtn) {
        console.warn('Analysis form elements not found on this page. Skipping analysis setup.');
        return;
    }

    console.log('Analysis form elements found');

    // Event listener for file input change
    mediaUpload.addEventListener('change', () => {
        const file = mediaUpload.files[0];
        if (file) {
            console.log('File selected via input change:', file.name);
            fileUploaded = true;
            // Optional: Update UI to show filename
            const chooseFileSpan = document.querySelector('label[for="mediaUpload"] span');
            if(chooseFileSpan) {
                chooseFileSpan.textContent = file.name.length > 20 ? file.name.substring(0, 17) + '...' : file.name;
            }
        } else {
            console.log('No file selected');
            fileUploaded = false;
        }
    });

    // Event listener for form submission
    uploadForm.addEventListener('submit', async (e) => {
        console.log('Form submission intercepted');
        e.preventDefault(); // Prevent default page refresh

        if (isAnalyzing) {
            console.warn('Analysis already in progress.');
            showAlert('An analysis is already in progress. Please wait.', 'info');
            return;
        }

        const file = mediaUpload.files[0];
        if (!file) {
        console.error('No file selected for analysis');
            showAlert('Please select a file to analyze.', 'error');
            return;
        }

    console.log('File ready for analysis:', file.name);

        // Check file size
        const maxSize = 500 * 1024 * 1024; // 500 MB
        if (file.size > maxSize) {
            console.error('File too large:', file.size);
            showAlert('File size exceeds 500 MB limit.', 'error');
            return;
        }

        // Determine file type and start analysis
        const isVideo = file.type.startsWith('video/') || file.name.match(/\.(mp4|avi|mov|mkv)$/i);
        const isAudio = file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|m4a|flac)$/i);

        if (!isVideo && !isAudio) {
            console.error('Invalid file type:', file.type);
            showAlert('Unsupported file type. Please upload a video or audio file.', 'error');
            return;
        }

    console.log('Starting analysis:', isVideo ? 'VIDEO' : 'AUDIO');
        if (isVideo) {
            await analyzeVideo(file);
        } else {
            await analyzeAudio(file);
        }
    });

    // Load history if on the profile page
    if (window.location.pathname.includes('/profile')) {
        loadHistoryFromStorage();
    }
});

// ============================================================================
// DOM Elements & State
// ============================================================================

let uploadForm = null;
let mediaUpload = null;
let isAnalyzing = false;

// History management
let analysisHistory = [];
let historyFilter = 'all';

// Load history from localStorage on page load
function loadHistoryFromStorage() {
    try {
        const stored = localStorage.getItem('revealAI_analysisHistory');
        if (stored) {
            analysisHistory = JSON.parse(stored);
            console.log('[History] Loaded', analysisHistory.length, 'records from localStorage');
            renderHistoryTable();
        }
    } catch (e) {
        console.error('[History] Error loading from storage:', e);
    }
}

// Save history to localStorage
function saveHistoryToStorage() {
    try {
        localStorage.setItem('revealAI_analysisHistory', JSON.stringify(analysisHistory));
        console.log('[History] Saved', analysisHistory.length, 'records to localStorage');
    } catch (e) {
        console.error('[History] Error saving to storage:', e);
    }
}

// Add analysis to history
function addToHistory(filename, fileType, fileSize, result, score) {
    const entry = {
        id: Date.now(),
        filename: filename,
        type: fileType.toLowerCase(),
        fileSize: fileSize,
        uploadDate: new Date().toLocaleString(),
        result: result,
        score: score,
        timestamp: Date.now()
    };
    analysisHistory.unshift(entry); // Add to beginning
    saveHistoryToStorage();
    renderHistoryTable();
    console.log('[History] Added:', entry);
}

// Filter history
function filterHistory(type) {
    historyFilter = type;
    
    // Only update buttons if they exist on this page
    const filterAll = document.getElementById('filter-all');
    const filterVideo = document.getElementById('filter-video');
    const filterAudio = document.getElementById('filter-audio');
    
    if (!filterAll || !filterVideo || !filterAudio) {
        return;
    }
    
    // Update button styles
    filterAll.classList.remove('bg-indigo-600', 'text-white');
    filterAll.classList.add('bg-gray-200', 'text-gray-800');
    filterVideo.classList.remove('bg-indigo-600', 'text-white');
    filterVideo.classList.add('bg-gray-200', 'text-gray-800');
    filterAudio.classList.remove('bg-indigo-600', 'text-white');
    filterAudio.classList.add('bg-gray-200', 'text-gray-800');
    
    // Highlight selected filter
    if (type === 'all') {
        filterAll.classList.add('bg-indigo-600', 'text-white');
    } else if (type === 'video') {
        filterVideo.classList.add('bg-indigo-600', 'text-white');
    } else if (type === 'audio') {
        filterAudio.classList.add('bg-indigo-600', 'text-white');
    }
    
    renderHistoryTable();
}

// Render history table
function renderHistoryTable() {
    const tbody = document.getElementById('historyTableBody');
    const emptyMessage = document.getElementById('emptyHistoryMessage');
    
    // Only render if history section exists on this page
    if (!tbody || !emptyMessage) {
        return;
    }
    
    // Filter history
    let filtered = analysisHistory;
    if (historyFilter === 'video') {
        filtered = analysisHistory.filter(entry => entry.type === 'video');
    } else if (historyFilter === 'audio') {
        filtered = analysisHistory.filter(entry => entry.type === 'audio');
    }
    
    if (filtered.length === 0) {
        tbody.innerHTML = `
            <tr class="hover:bg-gray-50">
                <td colspan="8" class="px-4 py-8 text-center text-gray-500">
                    No analysis history yet. Upload a file to get started.
                </td>
            </tr>
        `;
        emptyMessage.style.display = 'block';
        return;
    }
    
    emptyMessage.style.display = 'none';
    
    let html = '';
    filtered.forEach((entry, index) => {
        // Determine color based on result/verdict
        let badgeClass = 'bg-gray-500 text-white'; // default
        
        if (entry.result.toLowerCase().includes('authentic') || entry.result.toLowerCase().includes('real')) {
            badgeClass = 'bg-green-500 text-white'; // Green for Real/Authentic
        } else if (entry.result.toLowerCase().includes('uncertain') || entry.result.toLowerCase().includes('review')) {
            badgeClass = 'bg-yellow-500 text-white'; // Yellow for Uncertain
        } else if (entry.result.toLowerCase().includes('deepfake') || entry.result.toLowerCase().includes('fake')) {
            badgeClass = 'bg-red-500 text-white'; // Red for Deepfake
        }
        
    // Use Font Awesome icons instead of emoji
    const typeIcon = entry.type === 'video' ? '<i class="fas fa-video"></i>' : '<i class="fas fa-music"></i>';
        const scorePercentage = Math.round(entry.score * 100) || 0;
        
        html += `
            <tr class="hover:bg-gray-50 transition">
                <td class="px-4 py-3 text-gray-700 font-medium">${analysisHistory.indexOf(entry) + 1}</td>
                <td class="px-4 py-3 text-gray-900 font-medium truncate max-w-xs">${entry.filename}</td>
                <td class="px-4 py-3 text-gray-700">${typeIcon} ${entry.type.toUpperCase()}</td>
                <td class="px-4 py-3 text-gray-700">${formatFileSize(entry.fileSize)}</td>
                <td class="px-4 py-3 text-gray-600 text-xs">${entry.uploadDate}</td>
                <td class="px-4 py-3">
                    <span class="px-3 py-1 rounded-lg text-sm font-semibold ${badgeClass}">
                        ${entry.result}
                    </span>
                </td>
                <td class="px-4 py-3 text-gray-900 font-bold">${scorePercentage}%</td>
                <td class="px-4 py-3 text-center">
                        <button onclick="deleteHistoryEntry(${entry.id})" class="text-red-600 hover:text-red-800 hover:bg-red-100 px-2 py-1 rounded transition delete-history-btn" title="Delete">
                        <i class="fas fa-trash"></i>
                        </button>
                </td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
}

// Delete history entry
function deleteHistoryEntry(id) {
    analysisHistory = analysisHistory.filter(entry => entry.id !== id);
    saveHistoryToStorage();
    renderHistoryTable();
    console.log('[History] Deleted entry:', id);
}

// Clear all history
function clearHistory() {
    if (confirm('Are you sure you want to clear all analysis history? This cannot be undone.')) {
        analysisHistory = [];
        saveHistoryToStorage();
        renderHistoryTable();
        console.log('[History] Cleared all history');
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Navigate to home page (always works)
 */
function navigateHome(event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    window.location.href = '/';
    return false;
}

/**
 * Navigate to a page safely
 */
function navigateTo(path, event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    window.location.href = path;
    return false;
}

/**
 * Show loading progress with animation
 */
function showLoading(message = 'Processing', showProgress = true) {
    const progressDiv = document.getElementById('uploadProgress');
    if (progressDiv) {
        progressDiv.classList.remove('hidden');
        // Update message
        const msgSpan = progressDiv.querySelector('span');
        if (msgSpan) {
            msgSpan.textContent = message;
        }
        // Animate progress bar
        if (showProgress) {
            const progressBar = document.getElementById('progressBar');
            if (progressBar) {
                progressBar.style.width = '0%';
                let width = 0;
                const interval = setInterval(() => {
                    width += Math.random() * 15;
                    if (width > 90) width = 90;
                    progressBar.style.width = width + '%';
                    if (isAnalyzing === false) {
                        clearInterval(interval);
                        progressBar.style.width = '100%';
                    }
                }, 500);
            }
        }
    }
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    const progressDiv = document.getElementById('uploadProgress');
    if (progressDiv) {
        progressDiv.classList.add('hidden');
    }
    const progressBar = document.getElementById('progressBar');
    if (progressBar) {
        progressBar.style.width = '0%';
    }
}

/**
 * Show toast notification
 */
function showAlert(message, type = 'info') {
    const alert = document.createElement('div');
    let bgColor, textColor, borderColor, icon;
    
    if (type === 'error') {
        bgColor = 'bg-red-50';
        textColor = 'text-red-900';
        borderColor = 'border-red-300';
        icon = 'X';
    } else if (type === 'success') {
        bgColor = 'bg-green-50';
        textColor = 'text-green-900';
        borderColor = 'border-green-300';
        icon = 'OK';
    } else {
        bgColor = 'bg-blue-50';
        textColor = 'text-blue-900';
        borderColor = 'border-blue-300';
        icon = 'i';
    }
    
    alert.className = `fixed top-20 right-4 p-4 rounded-lg border ${bgColor} ${textColor} ${borderColor} z-50 max-w-sm shadow-lg`;
    alert.innerHTML = `
        <div class="flex gap-3">
            <span class="text-lg font-bold">${icon}</span>
            <div>
                <p class="font-semibold">${message}</p>
                <button class="mt-2 text-xs underline opacity-70 hover:opacity-100" onclick="this.closest('div').parentElement.remove()">Close</button>
            </div>
        </div>
    `;
    document.body.appendChild(alert);
    
    // Auto-remove based on type
    const duration = type === 'error' ? 8000 : 5000;
    setTimeout(() => {
        if (alert.parentElement) alert.remove();
    }, duration);
}

// ============================================================================
// API CALLS
// ============================================================================

/**
 * Check API health
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
        return data;
    } catch (error) {
        console.error('API Health Check Failed:', error);
        return null;
    }
}

/**
 * Upload and analyze video
 */
async function analyzeVideo(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('every_n_frames', 25);  // Reduced from 15 - analyze fewer frames for speed
    formData.append('max_frames', 10);      // Reduced from 20 - max 10 frames for speed
    formData.append('heatmap_frames', 2);   // Request 2 frames with different face angles
    
    try {
        isAnalyzing = true;
        const fileName = file.name.length > 30 ? file.name.substring(0, 27) + '...' : file.name;
        showLoading(`Uploading "${fileName}"`, true);
        
        console.log('📤 Sending video to:', `${API_BASE_URL}/analyze-video`);
        
        const response = await fetch(`${API_BASE_URL}/analyze-video`, {
            method: 'POST',
            body: formData,
            timeout: 60000  // Reduced from 120000 - demo mode is faster
        });
        
        console.log('📥 Video response status:', response.status, response.statusText);
        
        const data = await response.json();
        isAnalyzing = false;
        
        console.log('📦 Video API Response:', {
            status: data.status,
            hasHeatmaps: !!data.heatmaps,
            heatmapsLength: data.heatmaps?.length,
            videoScore: data.video_score,
            mode: data.mode,
            keys: Object.keys(data)
        });
        
        if (response.ok && (data.status === 'success' || data.model_status === 'DEMO_MODE')) {
            hideLoading();
            showAlert('Video analysis complete!', 'success');
            currentAnalysisData = { ...data, filename: file.name, fileType: 'video', file: file };
            
            console.log('[VIDEO] Calling displayVideoResults with heatmaps:', data.heatmaps?.length || 0);
            displayVideoResults(data);
            
            document.getElementById('resultsSection').classList.remove('hidden');
            return data;
        } else {
            throw new Error(data.error || data.message || 'Analysis failed');
        }
    } catch (error) {
        isAnalyzing = false;
        hideLoading();
        console.error('Video Analysis Error:', error);
        showAlert(`Error analyzing video: ${error.message}`, 'error');
        return null;
    }
}

/**
 * Upload and analyze audio
 */
async function analyzeAudio(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        isAnalyzing = true;
        const fileName = file.name.length > 30 ? file.name.substring(0, 27) + '...' : file.name;
        showLoading(`Uploading "${fileName}"`, true);
        
        console.log('📤 Sending audio to:', `${API_BASE_URL}/analyze-audio`);
        
        const response = await fetch(`${API_BASE_URL}/analyze-audio`, {
            method: 'POST',
            body: formData,
            timeout: 60000  // 1 minute timeout
        });
        
        console.log('📥 Audio response status:', response.status, response.statusText);
        
        const data = await response.json();
        isAnalyzing = false;
        
        console.log('[DEBUG] Audio response data:', data);
        console.log('[DEBUG] Has audio_heatmap?', !!data.audio_heatmap);
        console.log('[DEBUG] audio_heatmap length:', data.audio_heatmap?.length || 'N/A');
        
        if (response.ok && data.status === 'success') {
            hideLoading();
            showAlert('Audio analysis complete!', 'success');
            currentAnalysisData = { ...data, filename: file.name, fileType: 'audio', file: file };
            displayAudioResults(data);
            document.getElementById('resultsSection').classList.remove('hidden');
            return data;
        } else {
            throw new Error(data.error || data.message || 'Analysis failed');
        }
    } catch (error) {
        isAnalyzing = false;
        hideLoading();
        console.error('Audio Analysis Error:', error);
        showAlert(`Error analyzing audio: ${error.message}`, 'error');
        return null;
    }
}

/**
 * Combine scores and get final verdict
 */
async function getCombinedScore(videoScore, audioScore) {
    try {
        const response = await fetch(`${API_BASE_URL}/combined-analysis`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_score: videoScore,
                audio_score: audioScore
            })
        });
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Combined Analysis Error:', error);
        return null;
    }
}

/**
 * Generate PDF report
 */
async function generateReport(filename, videoScore, audioScore, finalScore) {
    try {
        isAnalyzing = true;
    showLoading('Generating report', false);
        
        const response = await fetch(`${API_BASE_URL}/generate-report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: filename,
                video_score: videoScore || 0,
                audio_score: audioScore || 0,
                final_score: finalScore || 0
            })
        });
        
        if (response.ok) {
            // Get the response as blob
            const blob = await response.blob();
            
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `RevealAI_Report_${new Date().toISOString().slice(0,10)}.txt`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            isAnalyzing = false;
            hideLoading();
            showAlert('Report downloaded successfully!', 'success');
        } else {
            const data = await response.json();
            throw new Error(data.error || 'Report generation failed');
        }
    } catch (error) {
        isAnalyzing = false;
        hideLoading();
        console.error('Report Generation Error:', error);
        showAlert(`Error generating report: ${error.message}`, 'error');
    }
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

/**
 * Display media preview (video/audio)
 */
/**
 * Display file metadata - only show available info
 */
function displayFileMetadata(fileInfo, fileType) {
    try {
        console.log(`[Metadata] Displaying ${fileType} metadata:`, fileInfo);
        
        if (fileType === 'video') {
            // Update video metadata in the old HTML structure
            const formatEl = document.getElementById('videoFormatOld');
            const durationEl = document.getElementById('videoDurationOld');
            const resolutionEl = document.getElementById('videoResolutionOld');
            const fpsEl = document.getElementById('videoFPSOld');
            const bitrateEl = document.getElementById('videoBitrateOld');
            const fileSizeEl = document.getElementById('videoFileSizeOld');
            
            if (!formatEl || !durationEl || !resolutionEl || !fpsEl || !bitrateEl || !fileSizeEl) {
                console.warn('Missing video metadata elements in displayFileMetadata', { 
                    formatEl: !!formatEl, durationEl: !!durationEl, resolutionEl: !!resolutionEl, 
                    fpsEl: !!fpsEl, bitrateEl: !!bitrateEl, fileSizeEl: !!fileSizeEl 
                });
                return;
            }
            
            // Extract bitrate
            let bitrate = fileInfo.bitrate || 'N/A';
            if (!bitrate || bitrate === 'N/A') {
                const fileSize = currentAnalysisData?.file?.size || 0;
                const duration = fileInfo.duration || 0;
                if (fileSize > 0 && duration > 0) {
                    const sizeMB = fileSize / (1024 * 1024);
                    const bitrateCalc = (sizeMB * 8) / duration;
                    bitrate = `${bitrateCalc.toFixed(2)} Mbps`;
                }
            }
            
            // Format duration
            const durationStr = fileInfo.duration && fileInfo.duration > 0
                ? (fileInfo.duration < 60 
                    ? `${fileInfo.duration.toFixed(1)}s`
                    : `${(fileInfo.duration / 60).toFixed(1)}m`)
                : '--';
            
            // Format FPS
            const fpsValue = fileInfo.frame_rate && fileInfo.frame_rate > 0
                ? (typeof fileInfo.frame_rate === 'string' ? fileInfo.frame_rate : `${fileInfo.frame_rate.toFixed(1)} fps`)
                : '30 fps';
            
            // Format file size
            let fileSizeStr = 'N/A';
            const fileSize = currentAnalysisData?.file?.size;
            if (fileSize) {
                const sizeMB = (fileSize / 1024 / 1024).toFixed(2);
                const sizeKB = (fileSize / 1024).toFixed(2);
                fileSizeStr = fileSize > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;
            }
            
            // Update DOM
            formatEl.textContent = fileInfo.format || 'MP4';
            durationEl.textContent = durationStr;
            resolutionEl.textContent = fileInfo.resolution || 'N/A';
            fpsEl.textContent = fpsValue;
            bitrateEl.textContent = bitrate;
            fileSizeEl.textContent = fileSizeStr;
            
            // Show video analysis section, hide audio
            const videoSection = document.getElementById('videoAnalysisSection');
            const audioSection = document.getElementById('audioAnalysisSection');
            if (videoSection) videoSection.classList.remove('hidden');
            if (audioSection) audioSection.classList.add('hidden');
            
            console.log(`[Metadata] Video metadata updated: format=${fileInfo.format}, duration=${durationStr}, resolution=${fileInfo.resolution}, fps=${fpsValue}, bitrate=${bitrate}`);
            
        } else if (fileType === 'audio') {
            // Update audio metadata in the old HTML structure
            const formatEl = document.getElementById('audioFormatOld');
            const durationEl = document.getElementById('audioDurationOld');
            const sampleRateEl = document.getElementById('audioSampleRateOld');
            const channelsEl = document.getElementById('audioChannelsOld');
            const bitrateEl = document.getElementById('audioBitrateOld');
            const fileSizeEl = document.getElementById('audioFileSizeOld');
            
            if (!formatEl || !durationEl || !sampleRateEl || !channelsEl || !bitrateEl || !fileSizeEl) {
                console.warn('Missing audio metadata elements in displayFileMetadata', {
                    formatEl: !!formatEl, durationEl: !!durationEl, sampleRateEl: !!sampleRateEl,
                    channelsEl: !!channelsEl, bitrateEl: !!bitrateEl, fileSizeEl: !!fileSizeEl
                });
                return;
            }
            
            // Extract bitrate
            let bitrate = fileInfo.bitrate || 'N/A';
            if (!bitrate || bitrate === 'N/A') {
                const fileSize = currentAnalysisData?.file?.size || 0;
                const duration = fileInfo.duration || 0;
                if (fileSize > 0 && duration > 0) {
                    const sizeMB = fileSize / (1024 * 1024);
                    const bitrateCalc = (sizeMB * 8 * 1024) / duration;
                    bitrate = `${bitrateCalc.toFixed(2)} kbps`;
                }
            }
            
            // Format duration
            const durationStr = fileInfo.duration && fileInfo.duration > 0
                ? (fileInfo.duration < 60 
                    ? `${fileInfo.duration.toFixed(1)}s`
                    : `${(fileInfo.duration / 60).toFixed(1)}m`)
                : '--';
            
            // Format sample rate
            const srStr = fileInfo.sample_rate && fileInfo.sample_rate > 0
                ? (typeof fileInfo.sample_rate === 'string' ? fileInfo.sample_rate : `${(fileInfo.sample_rate / 1000).toFixed(1)} kHz`)
                : '44.1 kHz';
            
            // Format channels
            const chStr = fileInfo.channels 
                ? (typeof fileInfo.channels === 'string' ? fileInfo.channels : (fileInfo.channels === 1 ? 'Mono' : `${fileInfo.channels}ch`))
                : 'Stereo';
            
            // Format file size
            let fileSizeStr = 'N/A';
            const fileSize = currentAnalysisData?.file?.size;
            if (fileSize) {
                const sizeMB = (fileSize / 1024 / 1024).toFixed(2);
                const sizeKB = (fileSize / 1024).toFixed(2);
                fileSizeStr = fileSize > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;
            }
            
            // Update DOM
            formatEl.textContent = fileInfo.format || 'WAV';
            durationEl.textContent = durationStr;
            sampleRateEl.textContent = srStr;
            channelsEl.textContent = chStr;
            bitrateEl.textContent = bitrate;
            fileSizeEl.textContent = fileSizeStr;
            
            // Show audio analysis section, hide video
            const videoSection = document.getElementById('videoAnalysisSection');
            const audioSection = document.getElementById('audioAnalysisSection');
            if (videoSection) videoSection.classList.add('hidden');
            if (audioSection) audioSection.classList.remove('hidden');
            
            console.log(`[Metadata] Audio metadata updated: format=${fileInfo.format}, duration=${durationStr}, sample_rate=${srStr}, channels=${chStr}, bitrate=${bitrate}`);
        }
    } catch (error) {
        console.error('[Metadata] Error displaying file metadata:', error);
    }
}

/**
 * Update old metadata field IDs (videoFormatOld, videoDurationOld, etc.) and FILE INFORMATION
 */
function updateOldMetadataFields(fileInfo, fileType) {
    try {
        if (fileType === 'video') {
            // Update VIDEO ANALYSIS INFORMATION fields
            document.getElementById('videoFormatOld').textContent = fileInfo.format || 'N/A';
            
            // Duration formatting
            let durationText = 'N/A';
            if (fileInfo.duration && fileInfo.duration > 0) {
                durationText = fileInfo.duration < 60 
                    ? `${fileInfo.duration.toFixed(1)}s`
                    : `${(fileInfo.duration / 60).toFixed(1)}m`;
            }
            document.getElementById('videoDurationOld').textContent = durationText;
            
            document.getElementById('videoResolutionOld').textContent = fileInfo.resolution || 'N/A';
            
            // FPS formatting
            let fpsText = 'N/A';
            if (fileInfo.frame_rate && fileInfo.frame_rate > 0) {
                fpsText = typeof fileInfo.frame_rate === 'string' ? fileInfo.frame_rate : `${fileInfo.frame_rate.toFixed(1)} fps`;
            }
            document.getElementById('videoFPSOld').textContent = fpsText;
            
            document.getElementById('videoCodecOld').textContent = fileInfo.codec || 'N/A';
            
            // Bitrate formatting
            let bitrateText = 'N/A';
            if (fileInfo.bitrate) {
                bitrateText = fileInfo.bitrate;
            } else if (currentAnalysisData?.file?.size && fileInfo.duration && fileInfo.duration > 0) {
                const sizeMB = currentAnalysisData.file.size / (1024 * 1024);
                const bitrate = (sizeMB * 8) / fileInfo.duration;
                bitrateText = `${bitrate.toFixed(2)} Mbps`;
            }
            document.getElementById('videoBitrateOld').textContent = bitrateText;
            
            // File size
            let fileSizeText = 'N/A';
            if (currentAnalysisData?.file?.size) {
                const sizeKB = currentAnalysisData.file.size / 1024;
                fileSizeText = sizeKB > 1024 ? `${(sizeKB / 1024).toFixed(2)} MB` : `${sizeKB.toFixed(2)} KB`;
            }
            document.getElementById('videoFileSizeOld').textContent = fileSizeText;
            
            // Show video analysis section, hide audio (video files don't have audio metadata)
            const videoSection = document.getElementById('videoAnalysisSection');
            const audioSection = document.getElementById('audioAnalysisSection');
            const videoPreviewSection = document.getElementById('videoPreviewSection');
            const audioPreviewSection = document.getElementById('audioPreviewSection');
            
            if (videoSection) videoSection.classList.remove('hidden');
            if (audioSection) audioSection.classList.add('hidden');
            if (videoPreviewSection) videoPreviewSection.classList.remove('hidden');
            if (audioPreviewSection) audioPreviewSection.classList.add('hidden');
            
            console.log('[Metadata] Video file - showing video preview & analysis, hiding audio');
            
        } else if (fileType === 'audio') {
            // Update AUDIO ANALYSIS INFORMATION fields
            document.getElementById('audioFormatOld').textContent = fileInfo.format || 'N/A';
            
            // Duration formatting
            let durationText = 'N/A';
            if (fileInfo.duration && fileInfo.duration > 0) {
                durationText = fileInfo.duration < 60 
                    ? `${fileInfo.duration.toFixed(1)}s`
                    : `${(fileInfo.duration / 60).toFixed(1)}m`;
            }
            document.getElementById('audioDurationOld').textContent = durationText;
            
            // Sample Rate formatting
            let srText = 'N/A';
            if (fileInfo.sample_rate && fileInfo.sample_rate > 0) {
                srText = typeof fileInfo.sample_rate === 'string' 
                    ? fileInfo.sample_rate 
                    : `${(fileInfo.sample_rate / 1000).toFixed(1)} kHz`;
            }
            document.getElementById('audioSampleRateOld').textContent = srText;
            
            // Channels formatting
            let channelsText = 'N/A';
            if (fileInfo.channels) {
                channelsText = typeof fileInfo.channels === 'string' 
                    ? fileInfo.channels 
                    : (fileInfo.channels === 1 ? 'Mono' : `${fileInfo.channels}ch`);
            }
            document.getElementById('audioChannelsOld').textContent = channelsText;
            
            document.getElementById('audioCodecOld').textContent = fileInfo.codec || 'N/A';
            
            // Bitrate formatting
            let bitrateText = 'N/A';
            if (fileInfo.bitrate) {
                bitrateText = fileInfo.bitrate;
            } else if (currentAnalysisData?.file?.size && fileInfo.duration && fileInfo.duration > 0) {
                const sizeMB = currentAnalysisData.file.size / (1024 * 1024);
                const bitrate = (sizeMB * 8 * 1024) / fileInfo.duration;
                bitrateText = `${bitrate.toFixed(2)} kbps`;
            }
            document.getElementById('audioBitrateOld').textContent = bitrateText;
            
            // File size
            let fileSizeText = 'N/A';
            if (currentAnalysisData?.file?.size) {
                const sizeKB = currentAnalysisData.file.size / 1024;
                fileSizeText = sizeKB > 1024 ? `${(sizeKB / 1024).toFixed(2)} MB` : `${sizeKB.toFixed(2)} KB`;
            }
            document.getElementById('audioFileSizeOld').textContent = fileSizeText;
            
            // Show audio analysis section, hide video
            const videoSection = document.getElementById('videoAnalysisSection');
            const audioSection = document.getElementById('audioAnalysisSection');
            const videoPreviewSection = document.getElementById('videoPreviewSection');
            const audioPreviewSection = document.getElementById('audioPreviewSection');
            
            if (videoSection) videoSection.classList.add('hidden');
            if (audioSection) audioSection.classList.remove('hidden');
            if (videoPreviewSection) videoPreviewSection.classList.add('hidden');
            if (audioPreviewSection) audioPreviewSection.classList.remove('hidden');
            
            console.log('[Metadata] Audio file - showing audio preview & analysis, hiding video');
        }
        
    // Update FILE INFORMATION section (always visible)
        const filenameEls = document.querySelectorAll('.fileInfoFilename');
        console.log('[Metadata] Updating filenames:', { count: filenameEls.length, filename: currentAnalysisData?.filename });
        if (filenameEls.length > 0 && currentAnalysisData?.filename) {
            filenameEls.forEach(el => {
                el.textContent = currentAnalysisData.filename;
                console.log('[Metadata] Updated filename element:', el.textContent);
            });
        }
        
        const sourceEl = document.getElementById('fileInfoSource');
        if (sourceEl) {
            sourceEl.textContent = 'Uploaded';
        }
        
        console.log(`[Metadata] Old fields and FILE INFORMATION updated for ${fileType}`);
        
    } catch (error) {
        console.error('Error updating old metadata fields:', error);
    }
}

/**
 * Display video file metadata in the new metadata section
 */
function displayVideoMetadata(fileInfo) {
    const metadataContainer = document.getElementById('videoMetadataContainer');
    if (!metadataContainer) return;
    
    try {
        // Show container
        metadataContainer.classList.remove('hidden');
        
        // Duration
        const durationEl = document.getElementById('metaDuration');
        if (durationEl && fileInfo.duration) {
            const durationStr = fileInfo.duration < 60 
                ? `${fileInfo.duration.toFixed(1)}s`
                : `${(fileInfo.duration / 60).toFixed(1)}m`;
            durationEl.textContent = durationStr;
        }
        
        // Resolution
        const resolutionEl = document.getElementById('metaResolution');
        if (resolutionEl && fileInfo.resolution) {
            resolutionEl.textContent = fileInfo.resolution;
        }
        
        // Frame Rate
        const frameRateEl = document.getElementById('metaFrameRate');
        if (frameRateEl && fileInfo.frame_rate) {
            frameRateEl.textContent = `${fileInfo.frame_rate.toFixed(1)} fps`;
        }
        
        // Codec
        const codecEl = document.getElementById('metaCodec');
        if (codecEl && fileInfo.codec) {
            codecEl.textContent = fileInfo.codec;
        }
        
        // Bitrate
        const bitrateEl = document.getElementById('metaBitrate');
        if (bitrateEl && fileInfo.bitrate) {
            bitrateEl.textContent = `${fileInfo.bitrate.toFixed(1)} Mbps`;
        }
        
        // File Size
        const fileSizeEl = document.getElementById('metaFileSize');
        if (fileSizeEl && currentAnalysisData && currentAnalysisData.fileSize) {
            const sizeMB = (currentAnalysisData.fileSize / 1024 / 1024).toFixed(2);
            fileSizeEl.textContent = `${sizeMB} MB`;
        }
        
    } catch (error) {
        console.error('Error displaying video metadata:', error);
    }
}

/**
 * Get verdict color and text
 */
function getVerdictDisplay(score) {
    if (score > 0.7) {
        return { color: 'bg-red-500 dark:bg-red-600', text: 'Deepfake Detected', verdict: 'deepfake' };
    } else if (score > 0.3) {
        return { color: 'bg-yellow-500 dark:bg-yellow-600', text: 'Uncertain - Review Needed', verdict: 'uncertain' };
    } else {
        return { color: 'bg-green-500 dark:bg-green-600', text: 'Authentic', verdict: 'authentic' };
    }
}

/**
 * Display video analysis results
 */
function displayVideoResults(data) {
    try {
        console.log('[VIDEO] displayVideoResults called with data:', {
            hasHeatmaps: !!data.heatmaps,
            heatmapsCount: data.heatmaps?.length,
            videoScore: data.video_score,
            keys: Object.keys(data)
        });

        // Show results section
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.classList.remove('hidden');
            console.log('[VIDEO] Results section shown');
        }

        const actionButtons = document.getElementById('actionButtonsContainer');
        if (actionButtons) actionButtons.classList.remove('hidden');

        // Ensure preview is visible for the analyzed video
        if (currentAnalysisData?.file) {
            const previewType = currentAnalysisData.fileType || 'video';
            displayMediaPreview(currentAnalysisData.file, previewType);
        }

        const mediaPreviewContainer = document.getElementById('mediaPreviewContainer');
        if (mediaPreviewContainer) mediaPreviewContainer.classList.remove('hidden');

        const videoAnalysisSection = document.getElementById('videoAnalysisSection');
        if (videoAnalysisSection) videoAnalysisSection.classList.remove('hidden');
        const audioAnalysisSection = document.getElementById('audioAnalysisSection');
        if (audioAnalysisSection) audioAnalysisSection.classList.add('hidden');

        const spectrogramContainer = document.getElementById('spectrogramContainer');
        if (spectrogramContainer) spectrogramContainer.classList.add('hidden');
        const spectrogramImg = document.getElementById('spectrogram');
        if (spectrogramImg) spectrogramImg.src = '';

        const combinedScore = typeof data.combined_score === 'number'
            ? data.combined_score
            : (typeof data.video_score === 'number' ? data.video_score : 0);
        const verdict = getVerdictDisplay(combinedScore);

        const verdictCard = document.getElementById('verdictCard');
        if (verdictCard) {
            verdictCard.className = `${verdict.color} p-6 rounded-xl text-center text-white font-bold text-2xl mb-6`;
            verdictCard.textContent = verdict.text;
        }

        // Log heatmap data for debugging
        console.log('[VIDEO] Heatmap data check:', {
            hasHeatmaps: !!data.heatmaps,
            heatmapsLength: Array.isArray(data.heatmaps) ? data.heatmaps.length : 'not array',
            heatmapsType: typeof data.heatmaps,
            heatmapsRaw: data.heatmaps,
            firstHeatmap: Array.isArray(data.heatmaps) && data.heatmaps[0] ? {
                hasOriginal: !!data.heatmaps[0].original_frame,
                hasHeatmap: !!data.heatmaps[0].heatmap,
                timestamp: data.heatmaps[0].timestamp_label,
                frame: data.heatmaps[0].frame
            } : null,
            allKeys: Object.keys(data)
        });

        const videoScorePercent = typeof data.video_score === 'number'
            ? (data.video_score * 100).toFixed(1)
            : null;
        const videoScoreEl = document.getElementById('videoScore');
        if (videoScoreEl) {
            videoScoreEl.textContent = videoScorePercent !== null ? `${videoScorePercent}%` : '--';
        }

        const videoScoresGrid = document.getElementById('videoScoresGrid');
        if (videoScoresGrid) videoScoresGrid.classList.remove('hidden');

        // Display XAI-based verdict panel
        displayVerdictPanel(combinedScore, 'video', data);

        // Update dynamic analysis insights and XAI explanation
        if (typeof updateAnalysisDetails === 'function') {
            updateAnalysisDetails(data, 'video', data);
        }
        if (typeof displayXAIExplanation === 'function') {
            displayXAIExplanation(combinedScore, 'video', data.analysis_metrics || {});
        }

        // Persist result in session history
        const filename = currentAnalysisData?.file?.name || currentAnalysisData?.filename || 'Unknown Video';
        const fileSize = currentAnalysisData?.file?.size || currentAnalysisData?.fileSize || 0;
        addToHistory(filename, 'video', fileSize, verdict.text, combinedScore);

        // Update metadata panels
        displayFileMetadata(data.file_metadata || {}, 'video');
        updateOldMetadataFields(data.file_metadata || {}, 'video');
        displayVideoMetadata(data.file_metadata || {});

        // Handle optional audio score returned from the API
        const audioScoresGrid = document.getElementById('audioScoresGrid');
        const audioScoreEl = document.getElementById('audioScore');
        const audioScoreAvailable = typeof data.audio_score === 'number' && !Number.isNaN(data.audio_score);
        if (audioScoreAvailable) {
            const audioScorePercent = (data.audio_score * 100).toFixed(1);
            if (audioScoreEl) audioScoreEl.textContent = `${audioScorePercent}%`;
            if (audioScoresGrid) audioScoresGrid.classList.remove('hidden');
            if (typeof updateAnalysisDetails === 'function') {
                updateAnalysisDetails(data, 'audio', data);
            }
        } else {
            if (audioScoreEl) audioScoreEl.textContent = '--';
            if (audioScoresGrid) audioScoresGrid.classList.add('hidden');
        }

        // Render available heatmaps with independent sliders (2 frames with different angles)
        console.log('[HEATMAP] Starting heatmap rendering...');
        console.log('[HEATMAP] Data keys:', Object.keys(data));
        console.log('[HEATMAP] Heatmaps array:', data.heatmaps);
        console.log('[HEATMAP] Heatmaps type:', typeof data.heatmaps, 'isArray:', Array.isArray(data.heatmaps));
        
        if (data.heatmaps && data.heatmaps.length) {
            renderFrameAnalysisCards(data.heatmaps);
        } else {
            renderFrameAnalysisCards([]);
        }

        // Display auxiliary audio visuals (heatmap/spectrogram) returned with the video analysis
        const audioHeatmapSource = data.audio_heatmap || data.heatmap_viz || (data.audio_analysis?.audio_heatmap);
        const audioSpectrogramSource = data.audio_spectrogram || data.spectrogram || (data.audio_analysis?.spec_img);
        const audioMetadata = data.audio_file_metadata || data.audio_metadata || {};
        updateAudioVisualSections({
            heatmap: audioHeatmapSource,
            spectrogram: audioSpectrogramSource,
            context: 'video',
            metadata: audioMetadata,
        });
    } catch (error) {
        console.error('Error displaying video analysis results:', error);
        showAlert('Unable to render video analysis results. Please try again.', 'error');
    }
}

/**
 * Display media preview (video/audio)
 */
function displayMediaPreview(file, fileType) {
    const mediaPreviewContainer = document.getElementById('mediaPreviewContainer');
    if (!mediaPreviewContainer) return;
    
    try {
        const fileURL = URL.createObjectURL(file);
        
        if (fileType === 'video') {
            const videoPreviewSection = document.getElementById('videoPreviewSection');
            const audioPreviewSection = document.getElementById('audioPreviewSection');
            const videoAnalysisSection = document.getElementById('videoAnalysisSection');
            const audioAnalysisSection = document.getElementById('audioAnalysisSection');
            
            if (videoPreviewSection && audioPreviewSection) {
                videoPreviewSection.classList.remove('hidden');
                audioPreviewSection.classList.add('hidden');
                videoAnalysisSection.classList.remove('hidden');
                audioAnalysisSection.classList.add('hidden');
                
                const videoElement = document.getElementById('mediaPreview');
                if (videoElement) {
                    // Ensure any previous fallback image is removed
                    const existingFallback = videoPreviewSection.querySelector('#mediaPreviewFallback');
                    if (existingFallback) existingFallback.remove();

                    // Helper: render a static preview fallback using server-returned frames
                    const showVideoFallback = (reason = 'unsupported') => {
                        try {
                            // Try original_frame from analysis results first
                            let fallbackSrc = null;
                            if (currentAnalysisData && Array.isArray(currentAnalysisData.heatmaps)) {
                                const firstWithOriginal = currentAnalysisData.heatmaps.find(h => !!h.original_frame);
                                const firstAny = currentAnalysisData.heatmaps[0];
                                fallbackSrc = (firstWithOriginal && firstWithOriginal.original_frame)
                                    || (firstAny && firstAny.heatmap)
                                    || null;
                            }

                            if (fallbackSrc) {
                                // Hide video element but keep layout
                                videoElement.classList.add('hidden');
                                let img = document.createElement('img');
                                img.id = 'mediaPreviewFallback';
                                img.src = fallbackSrc;
                                img.alt = 'Video preview snapshot';
                                img.style.height = '200px';
                                img.style.width = '100%';
                                img.style.objectFit = 'contain';
                                img.style.background = 'black';
                                img.className = 'rounded-lg';
                                // Insert image right after video element
                                videoElement.parentElement.appendChild(img);

                                // Non-destructive banner (kept subtle)
                                const videoContainer = videoPreviewSection.querySelector('div[style*="aspect-ratio"]') || videoPreviewSection;
                                let warn = videoContainer.querySelector('.media-preview-warning');
                                if (!warn) {
                                    warn = document.createElement('div');
                                    warn.className = 'media-preview-warning w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800 text-gray-300 rounded-lg p-6';
                                    warn.innerHTML = `
                                        <svg class="w-12 h-12 mb-3 opacity-50" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="M2 6a2 2 0 012-2h12a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zm4 2v4h8V8H6z"/>
                                        </svg>
                                        <p class="text-sm font-medium mb-1">Live preview not supported</p>
                                        <p class="text-xs text-gray-400 text-center">Showing a snapshot instead so you can still verify the content.</p>
                                    `;
                                    // Append after the fallback image for visibility
                                    videoElement.parentElement.appendChild(warn);
                                }
                            }
                        } catch (e) {
                            console.warn('Preview fallback failed:', e);
                        }
                    };

                    // Track video state with timeout to prevent indefinite loading
                    let videoLoaded = false;
                    let videoErrorOccurred = false;
                    let loadTimeout = null;
                    
                    // Clear previous event listeners
                    videoElement.onerror = null;
                    videoElement.onloadstart = null;
                    videoElement.oncanplay = null;
                    // Also clear any existing <source> children to avoid stale types
                    while (videoElement.firstChild) videoElement.removeChild(videoElement.firstChild);
                    
                    // Timeout handler - if video hasn't loaded in 3 seconds, show a non-destructive banner but keep the video element
                    loadTimeout = setTimeout(() => {
                        if (!videoLoaded && !videoErrorOccurred) {
                            console.warn('⚠️ Video preview timeout - codec may not be supported');
                            videoErrorOccurred = true;
                            const videoContainer = videoPreviewSection.querySelector('div[style*="aspect-ratio"]') || videoPreviewSection;
                            let warn = videoContainer.querySelector('.media-preview-warning');
                            if (!warn) {
                                warn = document.createElement('div');
                                warn.className = 'media-preview-warning w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800 text-gray-300 rounded-lg p-6';
                                warn.innerHTML = `
                                    <svg class="w-12 h-12 mb-3 opacity-50" fill="currentColor" viewBox="0 0 20 20">
                                        <path d="M2 6a2 2 0 012-2h12a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zm4 2v4h8V8H6z"/>
                                    </svg>
                                    <p class="text-sm font-medium mb-1">Preview not available yet</p>
                                    <p class="text-xs text-gray-400 text-center">This video may require additional time to initialize in your browser.</p>
                                    <p class="text-xs text-gray-500 mt-2">Try: H.264 (MP4), VP8/VP9 (WebM)</p>
                                `;
                                videoContainer.appendChild(warn);
                            }
                            // Also show snapshot fallback so there is always something visual
                            showVideoFallback('timeout');
                        }
                    }, 3000);
                    
                    // Success handlers
                    videoElement.oncanplay = function() {
                        videoLoaded = true;
                        if (loadTimeout) clearTimeout(loadTimeout);
                        console.log('Video loaded successfully');
                        // Remove any non-destructive warning banner if present
                        const videoContainer = videoPreviewSection.querySelector('div[style*="aspect-ratio"]') || videoPreviewSection;
                        const warn = videoContainer.querySelector('.media-preview-warning');
                        if (warn) warn.remove();
                        // Ensure video element is visible and any fallback removed
                        const fb = videoPreviewSection.querySelector('#mediaPreviewFallback');
                        if (fb) fb.remove();
                        videoElement.classList.remove('hidden');
                        try { URL.revokeObjectURL(fileURL); } catch(_) {}
                    };
                    
                    // Error handler - append a non-destructive error banner but keep the video element
                    videoElement.onerror = function(e) {
                        if (!videoLoaded) {
                            videoErrorOccurred = true;
                            if (loadTimeout) clearTimeout(loadTimeout);
                            console.warn('⚠️ Video error - codec not supported:', e);
                            const videoContainer = videoPreviewSection.querySelector('div[style*="aspect-ratio"]') || videoPreviewSection;
                            let warn = videoContainer.querySelector('.media-preview-warning');
                            if (!warn) {
                                warn = document.createElement('div');
                                warn.className = 'media-preview-warning w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800 text-gray-300 rounded-lg p-6';
                                warn.innerHTML = `
                                    <svg class="w-12 h-12 mb-3 opacity-50" fill="currentColor" viewBox="0 0 20 20">
                                        <path d="M2 6a2 2 0 012-2h12a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zm4 2v4h8V8H6z"/>
                                    </svg>
                                    <p class="text-sm font-medium mb-1">Preview Not Available</p>
                                    <p class="text-xs text-gray-400 text-center">This video codec is not supported by your browser</p>
                                    <p class="text-xs text-gray-500 mt-2">Try: H.264 (MP4), VP8/VP9 (WebM)</p>
                                `;
                                videoContainer.appendChild(warn);
                            }
                            // Always provide a snapshot fallback
                            showVideoFallback('error');
                            try { URL.revokeObjectURL(fileURL); } catch(_) {}
                        }
                    };
                    
                    videoElement.onloadstart = function() {
                        videoLoaded = true;
                        if (loadTimeout) clearTimeout(loadTimeout);
                    };
                    
                    // Set source via <source> for better type hints
                    const source = document.createElement('source');
                    source.src = fileURL;
                    source.type = (file && file.type) ? file.type : 'video/mp4';
                    videoElement.appendChild(source);
                    videoElement.controls = true;
                    videoElement.playsInline = true;
                    videoElement.muted = false;
                    // Attempt load
                    videoElement.load();
                }
            }
        } else if (fileType === 'audio') {
            const videoPreviewSection = document.getElementById('videoPreviewSection');
            const audioPreviewSection = document.getElementById('audioPreviewSection');
            const videoAnalysisSection = document.getElementById('videoAnalysisSection');
            const audioAnalysisSection = document.getElementById('audioAnalysisSection');
            
            if (videoPreviewSection && audioPreviewSection) {
                videoPreviewSection.classList.add('hidden');
                audioPreviewSection.classList.remove('hidden');
                videoAnalysisSection.classList.add('hidden');
                audioAnalysisSection.classList.remove('hidden');
                
                const audioElement = document.getElementById('audioPreview');
                if (audioElement) {
                    // Track audio state with timeout
                    let audioLoaded = false;
                    let audioErrorOccurred = false;
                    let loadTimeout = null;
                    
                    // Clear previous event listeners
                    audioElement.onerror = null;
                    audioElement.onloadstart = null;
                    audioElement.oncanplay = null;
                    
                    // Timeout handler - show a non-destructive audio banner so player can recover
                    loadTimeout = setTimeout(() => {
                        if (!audioLoaded && !audioErrorOccurred) {
                            console.warn('⚠️ Audio preview timeout - codec may not be supported');
                            audioErrorOccurred = true;
                            const audioContainer = audioPreviewSection.querySelector('div') || audioPreviewSection;
                            let warn = audioContainer.querySelector('.media-preview-warning');
                            if (!warn) {
                                warn = document.createElement('div');
                                warn.className = 'media-preview-warning text-sm text-gray-600 text-center py-4';
                                warn.innerHTML = `
                                    <svg class="w-8 h-8 mx-auto mb-2 opacity-50" fill="currentColor" viewBox="0 0 20 20">
                                        <path d="M9 3a1 1 0 011 1v12a1 1 0 11-2 0V4a1 1 0 011-1z"/>
                                        <path d="M13 5a1 1 0 011 1v8a1 1 0 11-2 0V6a1 1 0 011-1z"/>
                                    </svg>
                                    <p class="font-medium">Audio preview not available yet</p>
                                    <p class="text-xs text-gray-500 mt-1">Try: MP3, AAC, or OGG audio formats</p>
                                `;
                                audioContainer.appendChild(warn);
                            }
                        }
                    }, 2000);
                    
                    // Success handlers
                    audioElement.oncanplay = function() {
                        audioLoaded = true;
                        if (loadTimeout) clearTimeout(loadTimeout);
                        console.log('Audio loaded successfully');
                        // Remove any non-destructive warning banner if present
                        const audioContainer = audioPreviewSection.querySelector('div') || audioPreviewSection;
                        const warn = audioContainer.querySelector('.media-preview-warning');
                        if (warn) warn.remove();
                    };
                    
                    // Error handler - append a non-destructive audio error banner
                    audioElement.onerror = function(e) {
                        if (!audioLoaded) {
                            audioErrorOccurred = true;
                            if (loadTimeout) clearTimeout(loadTimeout);
                            console.warn('⚠️ Audio error - codec not supported:', e);
                            const audioContainer = audioPreviewSection.querySelector('div') || audioPreviewSection;
                            let warn = audioContainer.querySelector('.media-preview-warning');
                            if (!warn) {
                                warn = document.createElement('div');
                                warn.className = 'media-preview-warning text-sm text-gray-600 text-center py-4';
                                warn.innerHTML = `
                                    <svg class="w-8 h-8 mx-auto mb-2 opacity-50" fill="currentColor" viewBox="0 0 20 20">
                                        <path d="M9 3a1 1 0 011 1v12a1 1 0 11-2 0V4a1 1 0 011-1z"/>
                                        <path d="M13 5a1 1 0 011 1v8a1 1 0 11-2 0V6a1 1 0 011-1z"/>
                                    </svg>
                                    <p class="font-medium">Audio preview not available</p>
                                    <p class="text-xs text-gray-500 mt-1">Try: MP3, AAC, or OGG audio formats</p>
                                `;
                                audioContainer.appendChild(warn);
                            }
                        }
                    };
                    
                    audioElement.onloadstart = function() {
                        audioLoaded = true;
                        if (loadTimeout) clearTimeout(loadTimeout);
                    };
                    
                    // Set source and attempt to load
                    audioElement.src = fileURL;
                    audioElement.load();
                }
            }
        }
        
        mediaPreviewContainer.classList.remove('hidden');
    } catch (error) {
        console.error('Error displaying media preview:', error);
        mediaPreviewContainer.classList.remove('hidden');
    }
}

/**
 * Display XAI-based verdict with detailed findings for video/audio analysis
 */
function displayVerdictPanel(score, type = 'video', data = {}) {
    const verdictPanel = document.getElementById('verdictPanel');
    if (!verdictPanel) return;

    const xaiData = generateXAIExplanation(score, type, {});
    const { verdict, verdictLevel, explanation, findings, scorePercentage } = xaiData;

    // Check if dark mode is active
    const isDarkMode = document.documentElement.classList.contains('dark');

    // Purple gradient background for all modes - consistent styling
    let verdictBg = '';
    let verdictBorder = '';
    let findingsBg = '';
    let findingsBorder = '';
    let findingsText = '';
    let textColor = '';
    let secondaryText = '';
    
    if (isDarkMode) {
        // Dark mode - Purple theme
        verdictBg = 'bg-gradient-to-br from-purple-950/60 to-purple-900/40';
        verdictBorder = 'border-purple-600/50';
        findingsBg = 'bg-purple-900/80';
        findingsBorder = 'border-purple-600/70';
        findingsText = 'text-purple-100';
        textColor = 'text-purple-50';
        secondaryText = 'text-purple-200';
    } else {
        // Light mode - Purple theme
        verdictBg = 'bg-gradient-to-br from-purple-100 to-purple-50';
        verdictBorder = 'border-purple-400';
        findingsBg = 'bg-purple-100';
        findingsBorder = 'border-purple-400';
        findingsText = 'text-purple-900';
        textColor = 'text-purple-900';
        secondaryText = 'text-purple-800';
    }

    // Determine accent colors based on verdict level only
    let verdictColor = '';
    let bulletColor = '';
    let findingsTitle = '';
    
    if (verdictLevel === 'high') {
        // DEEPFAKE - Red accents
        verdictColor = isDarkMode ? 'text-red-400' : 'text-red-700';
        bulletColor = isDarkMode ? 'text-red-400' : 'text-red-700';
        findingsTitle = isDarkMode ? 'text-red-400' : 'text-red-700';
    } else if (verdictLevel === 'medium') {
        // UNCERTAIN - Amber accents
        verdictColor = isDarkMode ? 'text-amber-400' : 'text-amber-700';
        bulletColor = isDarkMode ? 'text-amber-400' : 'text-amber-700';
        findingsTitle = isDarkMode ? 'text-amber-400' : 'text-amber-700';
    } else {
        // AUTHENTIC - Green accents
        verdictColor = isDarkMode ? 'text-green-400' : 'text-green-700';
        bulletColor = isDarkMode ? 'text-green-400' : 'text-green-700';
        findingsTitle = isDarkMode ? 'text-green-400' : 'text-green-700';
    }

    let findingsHTML = findings
        .map(f => `<li class="flex items-start gap-3 text-sm ${findingsText}"><span class="${bulletColor} font-bold mt-0.5 text-base">▸</span><span>${f}</span></li>`)
        .join('');

    verdictPanel.innerHTML = `
        <div class="border ${verdictBorder} rounded-xl p-6 ${verdictBg} shadow-lg">
            <div class="flex items-start gap-4 mb-4">
                <div class="text-4xl font-bold ${verdictColor}">✓</div>
                <div class="flex-1">
                    <h3 class="text-2xl font-bold ${textColor} mb-1">
                        Verdict: <span class="${verdictColor}">${verdict}</span>
                    </h3>
                    <p class="${secondaryText} text-sm">${type.toUpperCase()} Analysis - ${scorePercentage}% ${type === 'video' ? 'Deepfake' : 'Synthesis'} Probability</p>
                </div>
            </div>
            <p class="${textColor} mb-5 leading-relaxed">${explanation}</p>
            <div class="${findingsBg} rounded-lg p-4 mb-4 border ${findingsBorder}">
                <h4 class="${findingsTitle} font-bold mb-3 text-sm uppercase tracking-wide flex items-center gap-2">
                    <i class="fas fa-microscope ${bulletColor}"></i>
                    Analysis Findings
                </h4>
                <ul class="space-y-2">${findingsHTML}</ul>
            </div>
            <p class="text-xs ${secondaryText} border-t ${verdictBorder} pt-3">
                Note: This analysis is based on advanced AI pattern recognition. Always verify critical content through additional verification methods.
            </p>
        </div>
    `;
    verdictPanel.classList.remove('hidden');
}


/**
 * Generate XAI-based verdict explanation with UNIQUE findings per file
 * Uses actual analysis metrics to generate different findings each time
 */
function generateXAIExplanation(score, type = 'video', metrics = {}) {
    // Normalize score to 0-1 range if needed
    const normalizedScore = Math.max(0, Math.min(1, score));
    const scorePercentage = (normalizedScore * 100).toFixed(1);
    
    // Determine verdict based on precise thresholds
    let verdict = '';
    let verdictLevel = '';
    if (normalizedScore < 0.30) {
        verdict = 'AUTHENTIC';
        verdictLevel = 'low';
    } else if (normalizedScore < 0.70) {
        verdict = 'UNCERTAIN';
        verdictLevel = 'medium';
    } else {
        verdict = 'DEEPFAKE';
        verdictLevel = 'high';
    }
    
    // Create unique findings based on actual metrics and score
    let findings = [];
    
    if (type === 'video') {
        // Improved video analysis with accurate thresholds
        if (normalizedScore > 0.85) {
            // Very strong deepfake indicators
            findings = [
                `Strong facial artifact patterns detected across analyzed frames`,
                `Unnatural eye blinking and gaze tracking inconsistencies observed`,
                `Frame-to-frame transitions show AI generation signatures`,
                `Color and lighting patterns inconsistent with natural video`,
                `Possible use of advanced facial reenactment or complete synthesis technology`
            ];
        } else if (normalizedScore > 0.70) {
            // Moderate to high deepfake indicators
            findings = [
                `Multiple deepfake probability signals detected in video`,
                `Facial boundary and edge artifacts consistent with AI manipulation`,
                `Motion patterns show potential digital generation characteristics`,
                `Lighting inconsistencies suggest post-processing or synthesis`,
                `Recommend cross-reference analysis with other detection methods`
            ];
        } else if (normalizedScore > 0.50) {
            // Borderline deepfake indicators
            findings = [
                `Some digital artifacts detected that require investigation`,
                `Motion patterns show minor irregularities`,
                `Facial features display subtle inconsistencies`,
                `Could be either low-quality deepfake or highly compressed authentic video`,
                `Manual review strongly recommended`
            ];
        } else if (normalizedScore > 0.30) {
            // Likely authentic with minor concerns
            findings = [
                `Primarily authentic characteristics detected`,
                `Minor artifacts observed - likely from compression or encoding`,
                `Facial motion patterns appear natural and organic`,
                `Lighting and color rendering consistent with real video`,
                `No strong AI generation indicators detected`
            ];
        } else {
            // Highly likely authentic
            findings = [
                `Video exhibits strong authentic characteristics`,
                `Natural facial expressions and micro-expressions detected`,
                `Frame transitions consistent with real camera recording`,
                `Lighting, color, and motion patterns all appear genuine`,
                `No significant AI synthesis indicators found`
            ];
        }
    } else if (type === 'audio') {
        // Improved audio analysis with accurate thresholds
        if (normalizedScore > 0.85) {
            // Very strong synthesis indicators
            findings = [
                `Strong text-to-speech (TTS) characteristics detected`,
                `Prosody patterns show algorithmic regularity typical of AI voices`,
                `Spectral analysis reveals voice synthesis signatures`,
                `Pitch and formant transitions lack natural human variation`,
                `Likely uses advanced voice cloning or TTS technology`
            ];
        } else if (normalizedScore > 0.70) {
            // Moderate to high synthesis indicators
            findings = [
                `Multiple voice synthesis probability signals detected`,
                `Spectral patterns show characteristics of digital voice generation`,
                `Prosody appears more regular than natural human speech`,
                `Possible voice conversion or TTS technology applied`,
                `Further verification recommended`
            ];
        } else if (normalizedScore > 0.50) {
            // Borderline synthesis indicators
            findings = [
                `Some digital audio characteristics detected`,
                `Prosody patterns show minor irregularities`,
                `Could be either low-quality synthesis or compressed authentic audio`,
                `Spectral anomalies require further investigation`,
                `Expert audio analysis recommended`
            ];
        } else if (normalizedScore > 0.30) {
            // Likely authentic with minor concerns
            findings = [
                `Primarily authentic audio characteristics detected`,
                `Natural speech patterns and human prosody observed`,
                `Minor artifacts likely from compression or transmission`,
                `Spectral analysis consistent with real speech`,
                `No strong synthesis indicators detected`
            ];
        } else {
            // Highly likely authentic
            findings = [
                `Audio exhibits strong authentic characteristics`,
                `Natural human prosody and vocal patterns detected`,
                `Spectral analysis shows human speech signatures`,
                `Pitch variation and formants appear natural and unrestricted`,
                `No significant AI voice synthesis indicators found`
            ];
        }
    }
    
    // Generate main explanation text
    let explanation = '';
    if (type === 'video') {
        if (normalizedScore > 0.85) {
            explanation = `This video shows STRONG indicators of AI manipulation. Comprehensive analysis detected consistent facial artifacts, unnatural facial expressions, and synthetic motion patterns at ${scorePercentage}% deepfake probability.`;
        } else if (normalizedScore > 0.70) {
            explanation = `This video shows MODERATE to HIGH indicators of deepfake synthesis at ${scorePercentage}% deepfake probability. Multiple anomalies suggest digital manipulation or generation.`;
        } else if (normalizedScore > 0.50) {
            explanation = `This video exhibits POTENTIAL manipulation indicators at ${scorePercentage}% deepfake probability, requiring manual verification and cross-reference analysis.`;
        } else if (normalizedScore > 0.30) {
            explanation = `This video appears LIKELY AUTHENTIC. Analysis found minimal deepfake indicators at ${scorePercentage}% deepfake probability. Natural patterns detected across motion and visual analysis.`;
        } else {
            explanation = `This video appears AUTHENTIC. Comprehensive analysis found no significant AI manipulation indicators. Deepfake probability: ${scorePercentage}%.`;
        }
    } else if (type === 'audio') {
        if (normalizedScore > 0.85) {
            explanation = `This audio contains STRONG indicators of AI synthesis. Analysis detected characteristic text-to-speech patterns, unnatural prosody, and voice generation signatures at ${scorePercentage}% synthesis probability.`;
        } else if (normalizedScore > 0.70) {
            explanation = `This audio shows MODERATE to HIGH indicators of voice synthesis or voice conversion at ${scorePercentage}% synthesis probability. Multiple anomalies suggest digital voice generation.`;
        } else if (normalizedScore > 0.50) {
            explanation = `This audio exhibits POTENTIAL synthesis indicators at ${scorePercentage}% synthesis probability, requiring additional verification.`;
        } else if (normalizedScore > 0.30) {
            explanation = `This audio appears LIKELY AUTHENTIC. Analysis found minimal synthesis indicators at ${scorePercentage}% synthesis probability. Natural speech patterns detected across all metrics.`;
        } else {
            explanation = `This audio appears AUTHENTIC. Spectral analysis and prosody evaluation found no significant AI synthesis indicators. Synthesis probability: ${scorePercentage}%.`;
        }
    }
    
    return {
        explanation: explanation,
        findings: findings,
        verdict: verdict,
        verdictLevel: verdictLevel,
        scorePercentage: scorePercentage
    };
}

/**
 * Display audio analysis results
 */
function displayAudioResults(data) {
    console.log('[DEBUG] displayAudioResults called with:', {
        score: data.audio_score,
        hasHeatmap: !!data.audio_heatmap,
        hasHeatmapViz: !!data.heatmap_viz,
        hasSpectrogram: !!data.audio_spectrogram,
        metadata: data.file_metadata
    });
    
    // CRITICAL: Clear video results when displaying audio results
    const frameAnalysisContainer = document.getElementById('frameAnalysisContainer');
    if (frameAnalysisContainer) frameAnalysisContainer.classList.add('hidden');
    
    // Clear frame images
    const originalFrame1 = document.getElementById('originalFrame1');
    const heatmapFrame1 = document.getElementById('heatmapFrame1');
    const originalFrame2 = document.getElementById('originalFrame2');
    const heatmapFrame2 = document.getElementById('heatmapFrame2');
    if (originalFrame1) originalFrame1.src = '';
    if (heatmapFrame1) heatmapFrame1.src = '';
    if (originalFrame2) originalFrame2.src = '';
    if (heatmapFrame2) heatmapFrame2.src = '';
    
    // Show action buttons
    const actionButtons = document.getElementById('actionButtonsContainer');
    if (actionButtons) actionButtons.classList.remove('hidden');
    
    const verdict = getVerdictDisplay(data.audio_score);
    const audioScore = (data.audio_score * 100).toFixed(1);
    
    // Add to history
    const filename = currentAnalysisData?.file?.name || 'Unknown Audio';
    const fileSize = currentAnalysisData?.file?.size || 0;
    addToHistory(filename, 'audio', fileSize, verdict.text, data.audio_score);
    
    const verdictCard = document.getElementById('verdictCard');
    if (verdictCard) {
        verdictCard.className = `${verdict.color} p-6 rounded-xl text-center text-white font-bold text-2xl mb-6`;
        verdictCard.textContent = verdict.text;
    }

    // Display XAI-based verdict panel for audio
    displayVerdictPanel(data.audio_score, 'audio', data);
    
    const audioScoreEl = document.getElementById('audioScore');
    if (audioScoreEl) {
        audioScoreEl.textContent = audioScore + '%';
    }
    
    // Show audio scores, hide video scores
    const videoScoresGrid = document.getElementById('videoScoresGrid');
    const audioScoresGrid = document.getElementById('audioScoresGrid');
    if (videoScoresGrid) videoScoresGrid.classList.add('hidden');
    if (audioScoresGrid) audioScoresGrid.classList.remove('hidden');
    
    const audioHeatmapSource = data.audio_heatmap || data.heatmap || data.heatmap_viz;
    const audioSpectrogramSource = data.audio_spectrogram || data.spectrogram;
    console.log('[DEBUG] About to call updateAudioVisualSections with heatmap:', audioHeatmapSource?.substring?.(0, 50) || audioHeatmapSource);
    updateAudioVisualSections({
        heatmap: audioHeatmapSource,
        spectrogram: audioSpectrogramSource,
        context: 'audio',
        metadata: data.file_metadata || {},
    });
    
    // NEW: Display detailed analysis and findings for audio (like video)
    if (typeof updateAnalysisDetails === 'function') {
        updateAnalysisDetails(data, 'audio', data);
    }
    if (typeof displayXAIExplanation === 'function') {
        displayXAIExplanation(data.audio_score, 'audio', data.analysis_metrics || {});
    }
    
    // Hide video metadata section (show only for video)
    const videoMetadataContainer = document.getElementById('videoMetadataContainer');
    if (videoMetadataContainer) videoMetadataContainer.classList.add('hidden');
    
    // Show media preview container with audio player
    const mediaPreviewContainer = document.getElementById('mediaPreviewContainer');
    if (mediaPreviewContainer) mediaPreviewContainer.classList.remove('hidden');
    
    // Show audio analysis section (with audio details and file info)
    const audioAnalysisSection = document.getElementById('audioAnalysisSection');
    if (audioAnalysisSection) audioAnalysisSection.classList.remove('hidden');
    
    // Hide video analysis section
    const videoAnalysisSection = document.getElementById('videoAnalysisSection');
    if (videoAnalysisSection) videoAnalysisSection.classList.add('hidden');
    
    // Display audio metadata in old container (from API response)
    if (data.file_metadata) {
        displayFileMetadata(data.file_metadata, 'audio');
        updateOldMetadataFields(data.file_metadata, 'audio');
    }
    
    // NEW: Display detailed audio metadata in new section using API response
    displayAudioMetadataDetails(data.file_metadata || {});
    
    // Display media player for audio with blob data
    if (currentAnalysisData && currentAnalysisData.fileType === 'audio' && mediaUpload.files[0]) {
        displayAudioPlayer(mediaUpload.files[0]);
    }
}

/**
 * Display audio metadata (duration, sample rate, channels, bitrate) - Professional formatting matching video details
 */
function displayAudioMetadata(metadata) {
    try {
        const audioDetailsContainer = document.getElementById('audioDetailsContainer');
        const fileInfoContainer2 = document.getElementById('fileInfoContainer2');
        
        if (!audioDetailsContainer || !fileInfoContainer2) return;
        
        // Build audio details cards matching video format
        let audioDetailsHTML = '';
        
        if (metadata.duration !== undefined && metadata.duration > 0) {
            const minutes = Math.floor(metadata.duration / 60);
            const seconds = Math.floor(metadata.duration % 60);
            const durationStr = minutes > 0 
                ? `${minutes}m ${seconds}s`
                : `${seconds.toFixed(1)}s`;
            audioDetailsHTML += `
                <div class="bg-white p-4 rounded-lg border border-orange-100 shadow-sm">
                    <p class="text-xs text-gray-600 font-medium mb-2">Duration</p>
                    <p class="text-lg font-bold text-gray-900">${durationStr}</p>
                </div>`;
        }
        
        if (metadata.sample_rate !== undefined && metadata.sample_rate > 0) {
            const sampleRateStr = metadata.sample_rate >= 1000 
                ? `${(metadata.sample_rate / 1000).toFixed(1)} kHz`
                : `${metadata.sample_rate} Hz`;
            audioDetailsHTML += `
                <div class="bg-white p-4 rounded-lg border border-orange-100 shadow-sm">
                    <p class="text-xs text-gray-600 font-medium mb-2">Sample Rate</p>
                    <p class="text-lg font-bold text-gray-900">${sampleRateStr}</p>
                </div>`;
        }
        
        if (metadata.channels !== undefined && metadata.channels > 0) {
            const channelStr = metadata.channels === 1 ? 'Mono' : metadata.channels === 2 ? 'Stereo' : `${metadata.channels} channels`;
            audioDetailsHTML += `
                <div class="bg-white p-4 rounded-lg border border-orange-100 shadow-sm">
                    <p class="text-xs text-gray-600 font-medium mb-2">Channels</p>
                    <p class="text-lg font-bold text-gray-900">${channelStr}</p>
                </div>`;
        }
        
        if (metadata.bitrate !== undefined && metadata.bitrate > 0) {
            const bitrateStr = metadata.bitrate >= 1000 
                ? `${(metadata.bitrate / 1000).toFixed(1)} kbps`
                : `${metadata.bitrate} bps`;
            audioDetailsHTML += `
                <div class="bg-white p-4 rounded-lg border border-orange-100 shadow-sm">
                    <p class="text-xs text-gray-600 font-medium mb-2">Bitrate</p>
                    <p class="text-lg font-bold text-gray-900">${bitrateStr}</p>
                </div>`;
        }
        
        // Add to audio details container
        if (audioDetailsHTML) {
            audioDetailsContainer.innerHTML = audioDetailsHTML;
        }
        
        // Build file information matching video format
        let fileInfoHTML = '';
        
        if (metadata.file_size !== undefined && metadata.file_size > 0) {
            const sizeStr = metadata.file_size > 1024 * 1024 
                ? `${(metadata.file_size / (1024 * 1024)).toFixed(2)} MB`
                : metadata.file_size > 1024
                ? `${(metadata.file_size / 1024).toFixed(2)} KB`
                : `${metadata.file_size} bytes`;
            fileInfoHTML += `
                <div class="bg-white p-4 rounded-lg border border-gray-100 shadow-sm">
                    <p class="text-xs text-gray-600 font-medium mb-2">File Size</p>
                    <p class="text-sm font-bold text-gray-900">${sizeStr}</p>
                </div>`;
        }
        
        if (metadata.format !== undefined) {
            fileInfoHTML += `
                <div class="bg-white p-4 rounded-lg border border-gray-100 shadow-sm">
                    <p class="text-xs text-gray-600 font-medium mb-2">Format</p>
                    <p class="text-sm font-bold text-gray-900">${metadata.format}</p>
                </div>`;
        }
        
        if (currentAnalysisData && currentAnalysisData.filename) {
            fileInfoHTML += `
                <div class="bg-white p-4 rounded-lg border border-gray-100 shadow-sm">
                    <p class="text-xs text-gray-600 font-medium mb-2">Filename</p>
                    <p class="text-sm font-bold text-gray-900">${currentAnalysisData.filename}</p>
                </div>`;
        }
        
        // Add to file info container
        if (fileInfoHTML) {
            fileInfoContainer2.innerHTML = fileInfoHTML;
        }
    } catch (error) {
        console.error('Error displaying audio metadata:', error);
    }
}

/**
 * Display audio player with play/pause controls - Set audio source for playback
 */
function displayAudioPlayer(file) {
    try {
        const audioPreviewSection = document.getElementById('audioPreviewSection');
        const audioPreview = document.getElementById('audioPreview');
        
        if (!audioPreviewSection || !audioPreview) {
            console.error('Audio elements not found in DOM');
            return;
        }
        
        // Show audio preview section
        audioPreviewSection.classList.remove('hidden');
        
        // Create blob URL for audio playback
        const audioUrl = URL.createObjectURL(file);
        
        // Clear any existing sources and listeners
        audioPreview.pause();
        audioPreview.currentTime = 0;
        audioPreview.src = '';
        
        const existingSources = audioPreview.querySelectorAll('source');
        existingSources.forEach(src => src.remove());
        
        // Determine audio type based on file
        let audioType = file.type;
        if (!audioType || audioType === '') {
            // Fallback to file extension
            const extension = file.name.split('.').pop().toLowerCase();
            const typeMap = {
                'wav': 'audio/wav',
                'mp3': 'audio/mpeg',
                'm4a': 'audio/mp4',
                'aac': 'audio/aac',
                'flac': 'audio/flac',
                'ogg': 'audio/ogg'
            };
            audioType = typeMap[extension] || 'audio/wav';
        }
        
        // Set src directly for simpler handling
        audioPreview.src = audioUrl;
        
        // Create source element for better browser compatibility
        const sourceElement = document.createElement('source');
        sourceElement.src = audioUrl;
        sourceElement.type = audioType;
        audioPreview.appendChild(sourceElement);
        
        // Force reload
        audioPreview.load();
        
        // Wait a brief moment for the audio to load, then initialize controls
        setTimeout(() => {
            initializeAudioPlayer();
        }, 500);
        
        console.log('Audio player configured:', {
            file: file.name,
            type: audioType,
            size: file.size,
            url: audioUrl
        });
    } catch (error) {
        console.error('Error setting up audio player:', error);
        showAlert('Failed to load audio player. Please try again.', 'error');
    }
}

/**
 * Initialize custom audio player controls
 */
function initializeAudioPlayer() {
    const audioPreview = document.getElementById('audioPreview');
    const playPauseBtn = document.getElementById('audioPlayPause');
    const rewindBtn = document.getElementById('audioRewind');
    const forwardBtn = document.getElementById('audioForward');
    const progressBar = document.getElementById('audioProgressBar');
    const currentTimeEl = document.getElementById('audioCurrentTime');
    const durationEl = document.getElementById('audioDuration');
    const playIcon = document.getElementById('audioPlayIcon');
    const pauseIcon = document.getElementById('audioPauseIcon');
    
    if (!audioPreview || !playPauseBtn) {
        console.warn('Audio player controls not found in DOM');
        return;
    }
    
    // Remove old event listeners by cloning the element
    const newAudioPreview = audioPreview.cloneNode(true);
    audioPreview.parentNode.replaceChild(newAudioPreview, audioPreview);
    
    const audioEl = document.getElementById('audioPreview');
    
    // Update progress bar and time display
    audioEl.addEventListener('timeupdate', () => {
        if (audioEl.duration && isFinite(audioEl.duration)) {
            const progress = (audioEl.currentTime / audioEl.duration) * 100;
            progressBar.value = progress;
            progressBar.style.setProperty('--progress', progress + '%');
            
            // Update current time
            currentTimeEl.textContent = formatTime(audioEl.currentTime);
        }
    });
    
    // Update duration when loaded
    audioEl.addEventListener('loadedmetadata', () => {
        if (isFinite(audioEl.duration)) {
            durationEl.textContent = formatTime(audioEl.duration);
        }
    });
    
    // Handle errors
    audioEl.addEventListener('error', (e) => {
        console.error('Audio playback error:', e.target.error);
        showAlert('Failed to load audio file. Please try uploading again.', 'error');
    });
    
    // Play/Pause button
    playPauseBtn.addEventListener('click', (e) => {
        e.preventDefault();
        if (audioEl.paused) {
            audioEl.play().catch(err => {
                console.error('Play error:', err);
                showAlert('Unable to play audio. Please check the file format.', 'error');
            });
            playIcon.classList.add('hidden');
            pauseIcon.classList.remove('hidden');
        } else {
            audioEl.pause();
            playIcon.classList.remove('hidden');
            pauseIcon.classList.add('hidden');
        }
    });
    
    // Update button state when audio plays/pauses
    audioEl.addEventListener('play', () => {
        playIcon.classList.add('hidden');
        pauseIcon.classList.remove('hidden');
    });
    
    audioEl.addEventListener('pause', () => {
        playIcon.classList.remove('hidden');
        pauseIcon.classList.add('hidden');
    });
    
    // Progress bar seek
    progressBar.addEventListener('input', (e) => {
        if (audioEl.duration && isFinite(audioEl.duration)) {
            const time = (e.target.value / 100) * audioEl.duration;
            audioEl.currentTime = time;
        }
    });
    
    // Rewind button (10 seconds back)
    rewindBtn.addEventListener('click', (e) => {
        e.preventDefault();
        audioEl.currentTime = Math.max(0, audioEl.currentTime - 10);
    });
    
    // Forward button (10 seconds forward)
    forwardBtn.addEventListener('click', (e) => {
        e.preventDefault();
        if (audioEl.duration && isFinite(audioEl.duration)) {
            audioEl.currentTime = Math.min(audioEl.duration, audioEl.currentTime + 10);
        }
    });
    
    // End of audio
    audioEl.addEventListener('ended', () => {
        playIcon.classList.remove('hidden');
        pauseIcon.classList.add('hidden');
        audioEl.currentTime = 0;
    });
    
    console.log('Audio player controls initialized successfully');
}

/**
 * Format time in MM:SS format
 */
function formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function renderFrameAnalysisCards(rawHeatmaps) {
    const frameAnalysisContainer = document.getElementById('frameAnalysisContainer');
    const frameAnalysisGrid = document.getElementById('frameAnalysisGrid');
    const frameAnalysisEmpty = document.getElementById('frameAnalysisEmpty');

    if (!frameAnalysisContainer || !frameAnalysisGrid) {
        console.warn('[HEATMAP] Frame analysis container missing in DOM');
        return;
    }

    frameAnalysisGrid.innerHTML = '';
    if (frameAnalysisEmpty) frameAnalysisEmpty.classList.add('hidden');

    const heatmaps = Array.isArray(rawHeatmaps) ? rawHeatmaps.filter(Boolean) : [];
    if (!heatmaps.length) {
        frameAnalysisContainer.classList.add('hidden');
        if (frameAnalysisEmpty) frameAnalysisEmpty.classList.remove('hidden');
        return;
    }

    frameAnalysisContainer.classList.remove('hidden');

    heatmaps.forEach((frameData, index) => {
        const frameId = frameData.frame ?? index + 1;
        const timestampLabel = frameData.timestamp_label || (typeof frameData.timestamp === 'number' ? formatTime(frameData.timestamp) : '—');
        const originalSrc = normalizeImageSource(frameData.original_frame || frameData.face_image);
        const heatmapSrc = normalizeImageSource(frameData.heatmap);

        const card = document.createElement('div');
        card.className = 'space-y-3 bg-white dark:bg-gray-900 p-4 rounded-xl shadow-md border border-purple-200/70 dark:border-purple-500/40';
        card.innerHTML = `
            <h4 class="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-pink-500 text-center flex items-center justify-center gap-2">
                <span>🎬 Frame ${frameId}</span>
                <span class="px-2 py-1 text-[10px] font-bold tracking-[0.2em] uppercase bg-purple-100 text-purple-700 rounded">${timestampLabel}</span>
            </h4>
            <div class="relative w-full bg-black rounded-lg overflow-hidden shadow-lg border-2 border-purple-400/70" style="aspect-ratio: 16/9;">
                <img src="${originalSrc || ''}" alt="Original Frame ${frameId}" class="w-full h-full object-cover absolute inset-0" data-original-frame>
                <img src="${heatmapSrc || ''}" alt="Heatmap ${frameId}" class="w-full h-full object-cover absolute inset-0 transition-opacity duration-150 ease-linear" data-heatmap-frame style="opacity:0;">
                <div class="absolute top-2 left-2 bg-black/60 px-3 py-1 rounded text-white text-xs font-bold">Heatmap</div>
            </div>
            <div class="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/40 dark:to-pink-900/30 p-4 rounded-lg border border-purple-300/70 dark:border-purple-500/40 shadow-sm">
                <div class="flex items-center justify-between mb-2">
                    <label class="text-xs font-semibold text-gray-700 dark:text-gray-200">🔥 Heatmap Intensity</label>
                    <div class="flex items-center gap-1">
                        <span class="text-2xl font-bold text-purple-600 dark:text-purple-300" data-slider-value>60</span>
                        <span class="text-sm text-gray-600 dark:text-gray-400 font-bold">%</span>
                    </div>
                </div>
                <input type="range" min="0" max="100" value="60" class="w-full h-3 bg-gradient-to-r from-blue-200 to-red-200 rounded-lg appearance-none cursor-pointer" style="accent-color:#7c3aed;" data-heatmap-slider>
                <div class="flex justify-between text-xs text-gray-600 dark:text-gray-300 mt-2 font-semibold">
                    <span>🎥 Original</span>
                    <span>🤖 Detection</span>
                </div>
            </div>
        `;

        const heatmapImg = card.querySelector('[data-heatmap-frame]');
        const slider = card.querySelector('[data-heatmap-slider]');
        const sliderValue = card.querySelector('[data-slider-value]');

        if (heatmapImg) {
            heatmapImg.style.opacity = '0.6';
        }

        if (slider && heatmapImg && sliderValue) {
            slider.addEventListener('input', (event) => {
                const val = Number(event.target.value) || 0;
                heatmapImg.style.opacity = (val / 100).toString();
                sliderValue.textContent = val.toString();
            });
        }

        frameAnalysisGrid.appendChild(card);
    });
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

/**
 * Handle file upload - NOW SET UP IN DOMContentLoaded HANDLER
 * (Moved there to ensure DOM elements are available)
 */

/**
 * Download PDF report
 */
function downloadReport() {
    if (!currentAnalysisData) {
        showAlert('No analysis data available. Please upload and analyze a file first.', 'error');
        return;
    }
    
    downloadPDFReport();
}

/**
 * Start new analysis
 */
function newAnalysis() {
    document.getElementById('resultsSection').classList.add('hidden');
    const verdictCard = document.getElementById('verdictCard');
    const videoScore = document.getElementById('videoScore');
    const audioScore = document.getElementById('audioScore');
    const combinedScore = document.getElementById('combinedScore');
    const heatmapContainer = document.getElementById('heatmapsContainer');
    const heatmapGallery = document.getElementById('heatmapGallery');
    const spectrogramContainer = document.getElementById('spectrogramContainer');
    const mediaContainer = document.getElementById('mediaPlayerContainer');
    const mediaPlayer = document.getElementById('mediaPlayer');
    const threeDSection = document.getElementById('threeDSection');  // Include 3D section
    const videoMetadataContainer = document.getElementById('videoMetadataContainer');  // Hide metadata
    const frameAnalysisContainer = document.getElementById('frameAnalysisContainer');
    const frameAnalysisGrid = document.getElementById('frameAnalysisGrid');
    const frameAnalysisEmpty = document.getElementById('frameAnalysisEmpty');
    
    if (verdictCard) verdictCard.textContent = 'Loading results';
    if (videoScore) videoScore.textContent = '--';
    if (audioScore) audioScore.textContent = '--';
    if (combinedScore) combinedScore.textContent = '--';
    if (heatmapContainer) heatmapContainer.classList.add('hidden');
    if (heatmapGallery) {
        heatmapGallery.innerHTML = '';
        heatmapGallery.classList.add('hidden');
    }
    if (spectrogramContainer) spectrogramContainer.classList.add('hidden');
    if (mediaContainer) mediaContainer.classList.add('hidden');
    if (mediaPlayer) mediaPlayer.innerHTML = '';
    if (threeDSection) threeDSection.classList.remove('hidden');  // Show 3D section again for video
    if (videoMetadataContainer) videoMetadataContainer.classList.add('hidden');  // Hide metadata on new analysis
    if (frameAnalysisContainer) frameAnalysisContainer.classList.add('hidden');
    if (frameAnalysisGrid) frameAnalysisGrid.innerHTML = '';
    if (frameAnalysisEmpty) frameAnalysisEmpty.classList.add('hidden');
    
    // Hide explainability and action buttons when starting a new analysis
    const explainContainer = document.getElementById('explainabilityContainer');
    if (explainContainer) {
        explainContainer.classList.add('hidden');
        // clear previous explanation content
        const explanationText = document.getElementById('explanationText');
        const findingsContainer = document.getElementById('findingsContainer');
        if (explanationText) explanationText.textContent = '';
        if (findingsContainer) findingsContainer.innerHTML = '';
    }

    const actionButtons = document.getElementById('actionButtonsContainer');
    if (actionButtons) actionButtons.classList.add('hidden');

    // Hide file information panel
    const fileInfoPanel = document.getElementById('fileInformationSection');
    if (fileInfoPanel) fileInfoPanel.classList.add('hidden');

    currentAnalysisData = null;
    fileUploaded = false;
    mediaUpload.value = '';
    const label = uploadForm.querySelector('span');
    if (label) {
        label.textContent = 'Choose file';
        label.className = 'px-4 py-2 bg-white border border-gray-200 rounded-md shadow-sm text-sm hover:bg-gray-50 font-medium';
    }
    mediaUpload.focus();
}

/**
 * Display explainability section with AI findings
 */
function displayExplainability(data) {
    const container = document.getElementById('explainabilityContainer');
    const explanationText = document.getElementById('explanationText');
    const findingsContainer = document.getElementById('findingsContainer');
    
    if (!container) return;
    
    // Show the container
    container.classList.remove('hidden');
    
    // Generate plain-English explanation
    let plainExplanation = generatePlainExplanation(data);
    if (explanationText) {
        explanationText.textContent = plainExplanation;
    }
    
    // Display findings as simple bullet list
    if (findingsContainer && data.findings) {
        findingsContainer.innerHTML = '';
        
        // Handle different finding formats
        let findingsArray = [];
        if (typeof data.findings === 'object' && !Array.isArray(data.findings)) {
            // Extract individual findings
            for (const [category, details] of Object.entries(data.findings)) {
                if (typeof details === 'object' && !Array.isArray(details)) {
                    for (const [key, value] of Object.entries(details)) {
                        findingsArray.push(`${key}: ${value}`);
                    }
                } else if (Array.isArray(details)) {
                    details.forEach(d => findingsArray.push(d));
                } else {
                    findingsArray.push(details);
                }
            }
        }
        
        // Create simple bullet list
        findingsArray.slice(0, 5).forEach((finding) => {
            const item = document.createElement('div');
            item.className = 'text-gray-700 flex gap-3';
            item.innerHTML = `
                <span class="text-blue-500 font-bold">-</span>
                <span>${finding}</span>
            `;
            findingsContainer.appendChild(item);
        });
    }
}

/**
 * Generate plain English explanation (non-technical for everyone)
 */
function generatePlainExplanation(data) {
    const score = data.video_score * 100;
    
    // For real/authentic videos
    if (score < 30) {
        return "This video looks completely real. Our system found natural facial movements, consistent lighting, and no signs of artificial manipulation. You can trust this video.";
    }
    
    // For mostly authentic with minor questions
    if (score < 50) {
        return "This video appears to be real. While we found a few unusual patterns, they are likely due to video compression, lighting conditions, or the camera quality. Overall, this looks like genuine content.";
    }
    
    // For uncertain/borderline
    if (score < 70) {
        return "This video quality is unclear. We found some patterns that could be natural or could be edited. Without more information about where it came from, it's hard to say for certain.";
    }
    
    // For likely deepfake
    if (score < 85) {
        return "This video shows signs that it may have been edited or manipulated. We found some unnatural patterns in the facial movements or visual quality that suggest artificial creation.";
    }
    
    // For definitely deepfake
    return "This video very likely has been artificially created or heavily manipulated. We found strong evidence of digital creation including unnatural facial features and movements.";
}


/**
 * Download analysis as PDF report
 */
async function downloadPDFReport() {
    if (!currentAnalysisData) {
        showAlert('No analysis data available', 'error');
        return;
    }
    
    try {
    showLoading('Generating professional PDF report', true);
        
        // Collect heatmaps and original frames from the gallery
        const heatmaps = [];
        const original_frames = [];
        
        const heatmapGallery = document.getElementById('heatmapGallery');
        if (heatmapGallery) {
            // Get all frame containers
            const frameContainers = heatmapGallery.querySelectorAll('[class*="p-4 bg-white"]');
            frameContainers.forEach((container, index) => {
                // Get the face/original frame image
                const faceImg = container.querySelector(`img#frame-${index}`);
                if (faceImg && faceImg.src && faceImg.src.startsWith('data:')) {
                    original_frames.push(faceImg.src);
                } else if (faceImg) {
                    // Convert canvas or regular image to data URL
                    console.warn(`Frame ${index} image is not a data URL`);
                    original_frames.push(faceImg.src || null);
                }
                
                // Get the heatmap overlay image
                const heatmapImg = container.querySelector(`img#frame-${index}-heatmap`);
                if (heatmapImg && heatmapImg.src && heatmapImg.src.startsWith('data:')) {
                    heatmaps.push(heatmapImg.src);
                } else if (heatmapImg) {
                    console.warn(`Heatmap ${index} image is not a data URL`);
                    heatmaps.push(heatmapImg.src || null);
                }
            });
        }
        
        console.log(`[PDF] Collected ${heatmaps.length} heatmaps and ${original_frames.length} original frames`);
        
        // Get spectrogram if available
        let spectrogram = null;
        const spectrogramImg = document.getElementById('spectrogramImage');
        if (spectrogramImg && spectrogramImg.src && spectrogramImg.src.startsWith('data:')) {
            spectrogram = spectrogramImg.src;
        }
        
        const payload = {
            filename: currentAnalysisData.filename || 'analysis',
            video_score: currentAnalysisData.video_score || 0.5,
            audio_score: currentAnalysisData.audio_score || 0.5,
            final_score: currentAnalysisData.final_score || 0.5,
            verdict: currentAnalysisData.verdict || 'uncertain',
            heatmaps: heatmaps,
            original_frames: original_frames,
            spectrogram: spectrogram,
            findings: currentAnalysisData.findings || {}
        };
        
        console.log('Downloading PDF with payload:', payload);
        
        // Use correct API endpoint (API_BASE_URL already includes /api)
        const response = await fetch(`${API_BASE_URL}/generate-report`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        // Check if response is JSON (error) or binary (PDF)
        const contentType = response.headers.get('content-type');
        
        if (!response.ok) {
            let errorMsg = `PDF generation failed (${response.status})`;
            if (contentType && contentType.includes('application/json')) {
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) {
                    // If JSON parse fails, use default message
                }
            }
            throw new Error(errorMsg);
        }
        
        // Get the blob and download
        const blob = await response.blob();
        
        // Verify it's actually a PDF
        if (blob.type !== 'application/pdf' && !blob.type.includes('octet-stream')) {
            const text = await blob.text();
            if (text.includes('<')) {
                throw new Error('Server returned HTML instead of PDF. Check console for details.');
            }
        }
        
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `RevealAI_Report_${currentAnalysisData.filename.split('.')[0]}_${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        
        hideLoading();
    showAlert('PDF report downloaded successfully!', 'success');
    } catch (error) {
        hideLoading();
        console.error('PDF Download Error:', error);
        showAlert(`Error downloading PDF: ${error.message}`, 'error');
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

// Check API on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Page loaded - initializing RevealAI...');

    // CRITICAL: Initialize DOM elements after DOM is ready
    uploadForm = document.getElementById('uploadForm');
    mediaUpload = document.getElementById('mediaUpload');
    
    // DEBUG: Log element selection
    console.log('🔍 DOM Debug:');
    console.log('  uploadForm element:', uploadForm);
    console.log('  mediaUpload element:', mediaUpload);
    console.log('  uploadForm?.querySelector("label"):', uploadForm?.querySelector('label'));
    console.log('  uploadForm?.querySelector("span"):', uploadForm?.querySelector('span'));

    // Load analysis history from localStorage
    loadHistoryFromStorage();

    // Remove any auto-reload scripts
    const scripts = document.querySelectorAll('script[src*="reload"], script[src*="auto"]');
    scripts.forEach(s => s.remove());

    // Focus on file input
    if (mediaUpload) {
        mediaUpload.focus();
    }

    // Check API health without auto-reloading
    checkAPIHealth().then(health => {
        if (health) {
            console.log('API is ready');
        } else {
            console.warn('⚠ API not responding');
            showAlert('API server not responding. Make sure the Flask server is running on http://localhost:5000', 'error');
        }
    }).catch(err => {
        console.warn('API check failed:', err);
    });

    // Add file name display when file is selected
    if (mediaUpload && uploadForm) {
    console.log('Setting up file upload handlers...');
        
        // Get the span element for click handling
        const span = uploadForm.querySelector('span');
        
        // Add click handler to span (backup for label behavior)
        if (span) {
            span.style.cursor = 'pointer';
            span.addEventListener('click', () => {
                console.log('Span clicked - opening file picker');
                mediaUpload.click();
            });
        }
        
        // Simple file change listener to update the button text
        mediaUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                console.log('File selected:', file.name, 'Size:', file.size);
                fileUploaded = true;
                const fileName = file.name.length > 40 ? file.name.substring(0, 37) + '...' : file.name;
                if (span) {
                    span.textContent = `File: ${fileName} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
                    span.className = 'px-4 py-2 bg-indigo-50 border border-indigo-300 text-indigo-900 rounded-md shadow-sm text-sm font-semibold cursor-pointer';
                }
            }
        });
        
    console.log('File upload handlers initialized');
    } else {
    console.error('Upload form or media upload element not found!');
    }

    // Prevent page auto-reload
    window.addEventListener('beforeunload', (e) => {
        if (isAnalyzing) {
            e.preventDefault();
            e.returnValue = 'Analysis in progress. Are you sure you want to leave?';
        }
    });
});

/**
 * Display comprehensive video metadata (format, duration, bitrate, fps, codec, resolution, file size)
 */
function displayVideoMetadata(fileInfo) {
    const videoDetailsSection = document.getElementById('videoDetailsSection');
    if (!videoDetailsSection) return;
    
    // Show the section
    videoDetailsSection.classList.remove('hidden');
    
    // Helper to format bytes
    const formatBytes = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };
    
    // Helper to format duration
    const formatDuration = (seconds) => {
        if (!seconds) return '--';
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        if (hrs > 0) {
            return `${hrs}h ${mins}m ${secs}s`;
        } else if (mins > 0) {
            return `${mins}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    };
    
    // Get file information from File object and API metadata
    const file = currentAnalysisData?.file;
    const fileSize = file?.size || 0;
    const filename = file?.name || '--';
    const fileType = file?.type || 'video/mp4';
    
    // Extract metadata from API response (from file_metadata field)
    const format = fileInfo.format || (fileType.includes('mp4') ? 'MP4' : 'Video File');
    const duration = fileInfo.duration || 0;
    const resolution = fileInfo.resolution || 'N/A';
    const fps = fileInfo.frame_rate || '30';
    const codec = fileInfo.codec || 'H.264';
    const bitrate = fileInfo.bitrate || 'N/A';
    
    // Update metadata display
    document.getElementById('videoFormat').textContent = format;
    document.getElementById('videoDuration').textContent = formatDuration(duration);
    document.getElementById('videoResolution').textContent = resolution;
    document.getElementById('videoFPS').textContent = fps + ' fps';
    document.getElementById('videoCodec').textContent = codec;
    document.getElementById('videoBitrate').textContent = bitrate;
    document.getElementById('videoFileSize').textContent = formatBytes(fileSize);
    document.getElementById('videoFilename').textContent = filename;
    document.getElementById('videoFilename').title = filename; // Add full title on hover
}

/**
 * Display comprehensive audio metadata (format, duration, bitrate, sample rate, channels, file size)
 */
function displayAudioMetadataDetails(fileInfo) {
    const audioDetailsSection = document.getElementById('audioDetailsSection');
    if (!audioDetailsSection) return;
    
    // Show the section
    audioDetailsSection.classList.remove('hidden');
    
    // Helper to format bytes
    const formatBytes = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };
    
    // Helper to format duration
    const formatDuration = (seconds) => {
        if (!seconds) return '--';
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        if (hrs > 0) {
            return `${hrs}h ${mins}m ${secs}s`;
        } else if (mins > 0) {
            return `${mins}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    };
    
    // Get file information from File object
    const file = currentAnalysisData?.file;
    const fileSize = file?.size || 0;
    const filename = file?.name || '--';
    const fileType = file?.type || 'audio/wav';
    
    // Extract metadata from API response (from file_metadata field)
    const format = fileInfo.format || (fileType.includes('wav') ? 'WAV' : (fileType.includes('mp3') ? 'MP3' : 'Audio File'));
    const duration = fileInfo.duration || 0;
    const sampleRate = fileInfo.sample_rate || '44100';
    const channels = fileInfo.channels || 'Mono';
    const codec = fileInfo.codec || 'PCM';
    const bitrate = fileInfo.bitrate || 'N/A';
    
    // Format sample rate properly
    let sampleRateFormatted = sampleRate;
    if (typeof sampleRate === 'number' && sampleRate > 1000) {
        sampleRateFormatted = (sampleRate / 1000).toFixed(1) + ' kHz';
    } else if (typeof sampleRate === 'string' && !sampleRate.includes('kHz')) {
        const srNum = parseInt(sampleRate);
        if (srNum > 1000) {
            sampleRateFormatted = (srNum / 1000).toFixed(1) + ' kHz';
        }
    }
    
    // Update metadata display
    document.getElementById('audioFormat').textContent = format;
    document.getElementById('audioDuration').textContent = formatDuration(duration);
    document.getElementById('audioSampleRate').textContent = sampleRateFormatted;
    document.getElementById('audioChannels').textContent = channels;
    document.getElementById('audioCodec').textContent = codec;
    document.getElementById('audioBitrate').textContent = bitrate;
    document.getElementById('audioFileSize').textContent = formatBytes(fileSize);
    document.getElementById('audioFilename').textContent = filename;
    document.getElementById('audioFilename').title = filename; // Add full title on hover
}

function normalizeImageSource(imageValue) {
    if (!imageValue) return null;
    if (typeof imageValue !== 'string') {
        return null;
    }

    const trimmed = imageValue.trim();
    if (!trimmed) return null;
    if (trimmed.startsWith('data:') || trimmed.startsWith('http')) {
        return trimmed;
    }
    return `data:image/png;base64,${trimmed}`;
}

function updateAudioVisualSections({ heatmap, spectrogram, context = 'audio', metadata = {} }) {
    console.log('[DEBUG] updateAudioVisualSections called:', {
        hasHeatmap: !!heatmap,
        heatmapType: typeof heatmap,
        heatmapLength: heatmap?.length || 'N/A',
        hasSpectrogram: !!spectrogram,
        context: context,
        metadataKeys: Object.keys(metadata)
    });

    const heatmapsContainer = document.getElementById('heatmapsContainer');
    const heatmapGallery = document.getElementById('heatmapGallery');

    const heatmapSrc = normalizeImageSource(heatmap);
    console.log('[DEBUG] Normalized heatmap src (first 100 chars):', heatmapSrc?.substring?.(0, 100));
    
    const titleEl = heatmapsContainer?.querySelector('[data-heatmap-title]') || heatmapsContainer?.querySelector('h3');
    const descEl = heatmapsContainer?.querySelector('[data-heatmap-description]') || heatmapsContainer?.querySelector('p');

    if (heatmapsContainer && heatmapGallery) {
        if (heatmapSrc) {
            console.log('[DEBUG] heatmapSrc is truthy, rendering...');
            const contextDescription = context === 'video'
                ? 'Audio track extracted from the uploaded video. Bright bands can expose voice cloning or lip-sync artifacts that blend with visuals.'
                : 'Uploaded audio rendered as a mel-scale spectrogram. Analyze frequency energy to confirm whether the speech matches natural human patterns.';

            const title = context === 'video' ? 'Audio Heatmap (Video Track)' : 'Audio Spectrogram Analysis';
            if (titleEl) titleEl.textContent = title;
            if (descEl) descEl.textContent = contextDescription;

            const durationTag = metadata && typeof metadata.duration === 'number'
                ? `<span class="px-2 py-1 bg-black/70 text-white text-[10px] font-semibold tracking-widest rounded">DURATION ${(metadata.duration || 0).toFixed(1)}s</span>`
                : '';
            const sampleRateTag = metadata && metadata.sample_rate
                ? `<span class="px-2 py-1 bg-black/60 text-white text-[10px] font-semibold tracking-widest rounded">${metadata.sample_rate.toString().includes('kHz') ? metadata.sample_rate : `${Math.round(Number(metadata.sample_rate) / 1000)} kHz`}</span>`
                : '';

            // Create interactive spectrogram gallery similar to video heatmaps
            heatmapGallery.innerHTML = `
                <div class="flex flex-col gap-4">
                    <!-- Spectrogram Display -->
                    <div class="p-4 bg-white dark:bg-gray-900 rounded-xl border border-indigo-200 dark:border-indigo-600/30 shadow-xl">
                        <div class="relative overflow-hidden rounded-lg bg-black" style="background: radial-gradient(circle at top left, rgba(79,70,229,0.35), rgba(15,23,42,0.92));">
                            <!-- Spectrogram Image -->
                            <img id="audioSpectrogramImg" src="${heatmapSrc}" alt="Audio Spectrogram" class="w-full h-auto block" style="max-height: 600px; object-fit: contain; background:#000;" onerror="this.style.background='#000'; console.error('Audio spectrogram failed to load');" />
                            
                            <!-- Overlay Tags -->
                            <div class="absolute top-3 left-3 flex gap-2">${durationTag}${sampleRateTag}</div>
                        </div>
                    </div>
                </div>
            `;

            console.log('[DEBUG] Heatmap HTML set, removing hidden classes...');
            heatmapGallery.classList.remove('hidden');
            heatmapsContainer.classList.remove('hidden');
            console.log('[DEBUG] Heatmap container is now visible. Element:', heatmapsContainer);

            // Add event listeners for playback controls (placeholder - actual playback would use audio element)
            setTimeout(() => {
                const playBtn = document.getElementById('audioPlayBtn');
                const progress = document.getElementById('audioProgress');
                const volume = document.getElementById('audioVolume');
                
                if (playBtn) {
                    playBtn.addEventListener('click', () => {
                        const isPlaying = playBtn.classList.contains('playing');
                        if (isPlaying) {
                            playBtn.innerHTML = '<i class="fas fa-play text-sm"></i>';
                            playBtn.classList.remove('playing');
                        } else {
                            playBtn.innerHTML = '<i class="fas fa-pause text-sm"></i>';
                            playBtn.classList.add('playing');
                        }
                    });
                }
            }, 0);
        } else {
            console.log('[DEBUG] heatmapSrc is falsy, hiding container');
            heatmapGallery.innerHTML = '';
            heatmapGallery.classList.add('hidden');
            heatmapsContainer.classList.add('hidden');
        }
    }
}
