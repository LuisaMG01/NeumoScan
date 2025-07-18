<template>
  <div class="upload-page">
    <!-- Animated Background -->
    <div class="animated-background">
      <div class="floating-particles">
        <div class="particle particle-1"></div>
        <div class="particle particle-2"></div>
        <div class="particle particle-3"></div>
        <div class="particle particle-4"></div>
        <div class="particle particle-5"></div>
        <div class="particle particle-6"></div>
      </div>
      <div class="grid-overlay"></div>
    </div>

    <div class="upload-container">
      <!-- Header Section -->
      <div class="header-section">
        <div class="status-indicator">
          <div class="pulse-ring"></div>
          <span>AI Analysis Ready</span>
        </div>
        
        <h1 class="page-title">
          <span class="title-gradient">X-Ray Analysis</span>
        </h1>
        
        <p class="page-subtitle">
          Upload your chest X-ray image and let our advanced AI detect pneumonia with 
          <span class="highlight">medical-grade precision</span>
        </p>
      </div>

      <!-- Main Upload Section -->
      <div class="upload-section">
        <div class="upload-card">
          <!-- File Upload Area -->
          <div class="upload-area" :class="{ 'has-file': selectedFile, 'drag-over': isDragOver }" 
               @drop.prevent="onDrop" 
               @dragover.prevent="isDragOver = true" 
               @dragleave.prevent="isDragOver = false">
            
            <div v-if="!selectedFile" class="upload-prompt">
              <div class="upload-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17,8 12,3 7,8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
              </div>
              <h3>Drop your X-ray here</h3>
              <p>or click to browse files</p>
              <div class="file-formats">
                <span>JPG</span>
                <span>PNG</span>
                <span>JPEG</span>
              </div>
            </div>

            <div v-if="selectedFile" class="file-preview">
              <div class="preview-container">
                <img :src="imagePreviewUrl" alt="X-ray preview" class="preview-image" />
                <div class="preview-overlay">
                  <button class="change-file-btn" @click="clearFile">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M3 6h18"/>
                      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>
                      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>
                    </svg>
                    Change File
                  </button>
                </div>
              </div>
              <div class="file-info">
                <h4>{{ selectedFile.name }}</h4>
                <p>{{ formatFileSize(selectedFile.size) }}</p>
              </div>
            </div>

            <input 
              ref="fileInput" 
              type="file" 
              @change="onFileChange" 
              accept="image/*" 
              class="file-input"
            />
          </div>

          <!-- Action Buttons -->
          <div class="action-buttons">
            <button v-if="!selectedFile" class="browse-btn" @click="$refs.fileInput.click()">
              <span class="btn-bg"></span>
              <span class="btn-content">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14,2 14,8 20,8"/>
                  <line x1="16" y1="13" x2="8" y2="13"/>
                  <line x1="16" y1="17" x2="8" y2="17"/>
                  <polyline points="10,9 9,9 8,9"/>
                </svg>
                Browse Files
              </span>
            </button>

            <button v-if="selectedFile" class="analyze-btn" @click="predict" :disabled="isAnalyzing">
              <span class="btn-bg"></span>
              <span class="btn-content">
                <svg v-if="!isAnalyzing" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M9 12l2 2 4-4"/>
                  <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"/>
                  <path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"/>
                  <path d="M21 12c0 1.66-1.34 3-3 3s-3-1.34-3-3"/>
                  <path d="M3 12c0 1.66 1.34 3 3 3s3-1.34 3-3"/>
                </svg>
                <div v-if="isAnalyzing" class="loading-spinner"></div>
                {{ isAnalyzing ? 'Analyzing...' : 'Analyze X-Ray' }}
              </span>
            </button>
          </div>
        </div>

        <!-- Results Section -->
        <div v-if="prediction" class="results-section">
          <div class="result-card">
            <div class="result-header">
              <div class="result-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
              </div>
              <h3>Analysis Complete</h3>
            </div>

            <div class="result-content">
              <div class="diagnosis-result">
                <label>Diagnosis:</label>
                <span class="diagnosis-value" :class="getDiagnosisClass(prediction.class)">
                  {{ formatDiagnosis(prediction.class) }}
                </span>
              </div>

              <div class="confidence-result">
                <label>Confidence Level:</label>
                <div class="confidence-bar">
                  <div class="confidence-fill" :style="{ width: (prediction.confidence * 100) + '%' }"></div>
                  <span class="confidence-text">{{ (prediction.confidence * 100).toFixed(1) }}%</span>
                </div>
              </div>

              <div class="result-details">
                <h4>Analysis Details</h4>
                <ul>
                  <li>
                    <span>Processing Time:</span>
                    <span>{{ processingTime }}ms</span>
                  </li>
                  <li>
                    <span>Model Version:</span>
                    <span>NeumoScan v2.1</span>
                  </li>
                  <li>
                    <span>Image Quality:</span>
                    <span class="quality-good">Excellent</span>
                  </li>
                </ul>
              </div>
            </div>

            <div class="result-actions">
              <button class="secondary-btn" @click="downloadReport">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7,10 12,15 17,10"/>
                  <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                Download Report
              </button>
              <button class="primary-btn" @click="analyzeAnother">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M1 4v6h6"/>
                  <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
                </svg>
                Analyze Another
              </button>
            </div>
          </div>
        </div>

        <!-- Similar Images -->
        <div class="similar-images-container" v-if="similarImages.length">
          <h3 class="title-gradient">Similar Cases</h3>
          <div class="medical-explanation-container" v-if="medicalExplanation">
            <h3 class="title-gradient">Medical Explanation</h3>
            <p class="medical-explanation-text">{{ medicalExplanation }}</p>
          </div>
          <div class="similar-images-grid">
            <div
              v-for="(img, index) in similarImages"
              :key="index"
              class="similar-card"
            >
              <div class="similar-image-wrapper">
                <img :src="'data:image/jpeg;base64,' + img.image_base64" :alt="'Similar image ' + index" />
                <div class="similar-overlay">
                  <span class="similarity-badge">{{ (img.similarity * 100).toFixed(1) }}%</span>
                </div>
              </div>
              <div class="similar-info">
                <div class="similar-diagnosis" :class="getDiagnosisClass(img.class)">
                  {{ formatDiagnosis(img.class) }}
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'UploadImage',
  data() {
    return {
      selectedFile: null,
      prediction: null,
      imagePreviewUrl: null,
      isDragOver: false,
      isAnalyzing: false,
      processingTime: 0,
      similarImages: []
    };
  },
  methods: {
    onFileChange(event) {
      const file = event.target.files[0];
      if (file) {
        this.handleFile(file);
      }
    },
    onDrop(event) {
      this.isDragOver = false;
      const file = event.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        this.handleFile(file);
      }
    },
    handleFile(file) {
      this.selectedFile = file;
      this.prediction = null;
      this.imagePreviewUrl = URL.createObjectURL(file);
    },
    clearFile() {
      this.selectedFile = null;
      this.prediction = null;
      this.imagePreviewUrl = null;
      this.$refs.fileInput.value = '';
    },
    async predict() {
      if (!this.selectedFile) return;

      this.isAnalyzing = true;
      const startTime = Date.now();
      
      const formData = new FormData();
      formData.append("file", this.selectedFile);

      try {
        const response = await fetch("http://localhost:8000/predict", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        this.prediction = data;
        this.similarImages = data.similar_images;
        this.processingTime = Date.now() - startTime;
      } catch (error) {
        console.error("Prediction error:", error);
      } finally {
        this.isAnalyzing = false;
      }
    },
    formatFileSize(bytes) {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    formatDiagnosis(diagnosis) {
      const formats = {
        'normal': 'Normal',
        'bacterial': 'Bacterial Pneumonia',
        'viral': 'Viral Pneumonia'
      };
      return formats[diagnosis] || diagnosis;
    },
    getDiagnosisClass(diagnosis) {
      return {
        'normal': 'diagnosis-normal',
        'bacterial': 'diagnosis-bacterial',
        'viral': 'diagnosis-viral'
      }[diagnosis] || 'diagnosis-unknown';
    },
    downloadReport() {
      // Implementation for downloading report
      console.log('Downloading report...');
    },
    analyzeAnother() {
      this.clearFile();
    }
  }
};
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

.upload-page {
  min-height: 100vh;
  background: #0a0a0a;
  color: white;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  overflow-x: hidden;
  position: relative;
}

/* Animated Background */
.animated-background {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 1;
}

.floating-particles {
  position: absolute;
  width: 100%;
  height: 100%;
}

.particle {
  position: absolute;
  border-radius: 50%;
  opacity: 0.4;
  animation: floatParticle 12s ease-in-out infinite;
}

.particle-1 {
  width: 60px;
  height: 60px;
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  top: 15%;
  left: 15%;
  animation-delay: 0s;
}

.particle-2 {
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #ff0080 0%, #ff4569 100%);
  top: 25%;
  right: 25%;
  animation-delay: -3s;
}

.particle-3 {
  width: 30px;
  height: 30px;
  background: linear-gradient(135deg, #00ff88 0%, #00cc69 100%);
  bottom: 35%;
  left: 35%;
  animation-delay: -6s;
}

.particle-4 {
  width: 50px;
  height: 50px;
  background: linear-gradient(135deg, #ffaa00 0%, #ff6600 100%);
  bottom: 15%;
  right: 15%;
  animation-delay: -9s;
}

.particle-5 {
  width: 35px;
  height: 35px;
  background: linear-gradient(135deg, #aa00ff 0%, #6600cc 100%);
  top: 65%;
  left: 65%;
  animation-delay: -2s;
}

.particle-6 {
  width: 45px;
  height: 45px;
  background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
  top: 45%;
  right: 45%;
  animation-delay: -5s;
}

@keyframes floatParticle {
  0%, 100% { 
    transform: translateY(0px) translateX(0px) rotate(0deg);
    opacity: 0.4;
  }
  33% { 
    transform: translateY(-30px) translateX(20px) rotate(120deg);
    opacity: 0.7;
  }
  66% { 
    transform: translateY(-15px) translateX(-15px) rotate(240deg);
    opacity: 0.3;
  }
}

.grid-overlay {
  position: absolute;
  width: 100%;
  height: 100%;
  background-image: 
    linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
  background-size: 60px 60px;
  animation: gridFloat 25s linear infinite;
}

@keyframes gridFloat {
  0% { transform: translate(0, 0); }
  100% { transform: translate(60px, 60px); }
}

/* Main Container */
.upload-container {
  position: relative;
  z-index: 2;
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  min-height: 100vh;
}

/* Header Section */
.header-section {
  text-align: center;
  margin-bottom: 4rem;
  padding-top: 2rem;
}

.status-indicator {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 50px;
  margin-bottom: 2rem;
  backdrop-filter: blur(20px);
  font-size: 0.9rem;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.8);
}

.pulse-ring {
  width: 8px;
  height: 8px;
  background: #00ff88;
  border-radius: 50%;
  animation: pulseRing 2s ease-in-out infinite;
}

@keyframes pulseRing {
  0%, 100% { 
    opacity: 1;
    transform: scale(1);
  }
  50% { 
    opacity: 0.5;
    transform: scale(1.3);
  }
}

.page-title {
  font-size: clamp(2.5rem, 6vw, 4rem);
  font-weight: 800;
  margin-bottom: 1.5rem;
  letter-spacing: -0.02em;
}

.title-gradient {
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 50%, #0044cc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: titleShift 4s ease-in-out infinite;
}

@keyframes titleShift {
  0%, 100% { filter: hue-rotate(0deg); }
  50% { filter: hue-rotate(30deg); }
}

.page-subtitle {
  font-size: 1.2rem;
  color: rgba(255, 255, 255, 0.7);
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
}

.page-subtitle .highlight {
  color: #00ff88;
  font-weight: 600;
}

/* Upload Section */
.upload-section {
  display: grid;
  grid-template-columns: 1fr;
  gap: 2rem;
  max-width: 800px;
  margin: 0 auto;
}

.upload-card {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 24px;
  padding: 3rem;
  backdrop-filter: blur(20px);
  transition: all 0.3s ease;
}

.upload-card:hover {
  background: rgba(255, 255, 255, 0.05);
  border-color: rgba(0, 245, 255, 0.3);
}

/* Upload Area */
.upload-area {
  border: 2px dashed rgba(255, 255, 255, 0.2);
  border-radius: 16px;
  padding: 3rem;
  text-align: center;
  transition: all 0.3s ease;
  cursor: pointer;
  margin-bottom: 2rem;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-area:hover {
  border-color: rgba(0, 245, 255, 0.5);
  background: rgba(0, 245, 255, 0.05);
}

.upload-area.drag-over {
  border-color: #00ff88;
  background: rgba(0, 255, 136, 0.1);
  transform: scale(1.02);
}

.upload-area.has-file {
  border-color: rgba(0, 245, 255, 0.5);
  background: rgba(0, 245, 255, 0.05);
}

.upload-prompt {
  width: 100%;
}

.upload-icon {
  width: 80px;
  height: 80px;
  margin: 0 auto 1.5rem;
  padding: 1.5rem;
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-icon svg {
  width: 100%;
  height: 100%;
  color: white;
}

.upload-prompt h3 {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: white;
}

.upload-prompt p {
  color: rgba(255, 255, 255, 0.6);
  margin-bottom: 1.5rem;
}

.file-formats {
  display: flex;
  justify-content: center;
  gap: 1rem;
}

.file-formats span {
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.8);
}

.file-input {
  display: none;
}

/* File Preview */
.file-preview {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.preview-container {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  max-width: 300px;
  width: 100%;
}

.preview-image {
  width: 100%;
  height: auto;
  display: block;
}

.preview-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.3s ease;
}

.preview-container:hover .preview-overlay {
  opacity: 1;
}

.change-file-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 30px;
  color: white;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);
}

.change-file-btn:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: translateY(-2px);
}

.change-file-btn svg {
  width: 16px;
  height: 16px;
}

.file-info {
  text-align: center;
}

.file-info h4 {
  font-size: 1.1rem;
  font-weight: 600;
  color: white;
  margin-bottom: 0.25rem;
}

.file-info p {
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.9rem;
}

/* Action Buttons */
.action-buttons {
  display: flex;
  justify-content: center;
  gap: 1rem;
}

.browse-btn,
.analyze-btn {
  position: relative;
  display: flex;
  align-items: center;
  padding: 1rem 2rem;
  border: none;
  border-radius: 50px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  overflow: hidden;
  background: transparent;
}

.btn-bg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border-radius: 50px;
  transition: all 0.3s ease;
}

.browse-btn .btn-bg {
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
}

.analyze-btn .btn-bg {
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
}

.browse-btn:hover .btn-bg {
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.1) 100%);
  transform: scale(1.05);
}

.analyze-btn:hover .btn-bg {
  background: linear-gradient(135deg, #00f5ff 0%, #0044cc 100%);
  transform: scale(1.05);
}

.analyze-btn:disabled .btn-bg {
  background: rgba(255, 255, 255, 0.1);
}

.analyze-btn:disabled {
  cursor: not-allowed;
}

.btn-content {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  z-index: 1;
  color: white;
}

.btn-content svg {
  width: 18px;
  height: 18px;
}

.loading-spinner {
  width: 18px;
  height: 18px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top: 2px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Results Section */
.results-section {
  margin-top: 2rem;
}

.result-card {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 24px;
  padding: 2.5rem;
  backdrop-filter: blur(20px);
  animation: slideUp 0.5s ease-out;
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.result-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 2rem;
}

.result-icon {
  width: 50px;
  height: 50px;
  background: linear-gradient(135deg, #00ff88 0%, #00cc69 100%);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.result-icon svg {
  width: 24px;
  height: 24px;
  color: white;
}

.result-header h3 {
  font-size: 1.5rem;
  font-weight: 700;
  color: white;
}

.result-content {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.diagnosis-result {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
}

.diagnosis-result label {
  font-weight: 600;
  color: rgba(255, 255, 255, 0.8);
  min-width: 100px;
}

.diagnosis-value {
  font-size: 1.1rem;
  font-weight: 700;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.diagnosis-normal {
  background: rgba(0, 255, 136, 0.2);
  color: #00ff88;
  border: 1px solid #00ff88;
}

.diagnosis-bacterial {
  background: rgba(255, 68, 68, 0.2);
  color: #ff4444;
  border: 1px solid #ff4444;
}

.diagnosis-viral {
  background: rgba(255, 184, 68, 0.2);
  color: #ffb844;
  border: 1px solid #ffb844;
}

.confidence-result {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.confidence-result label {
  font-weight: 600;
  color: rgba(255, 255, 255, 0.8);
}

.confidence-bar {
  position: relative;
  height: 12px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  background: linear-gradient(90deg, #00ff88 0%, #00cc69 100%);
  border-radius: 6px;
  transition: width 0.5s ease;
}

.confidence-text {
  position: absolute;
  right: 0.5rem;
  top: 50%;
  transform: translateY(-50%);
  font-size: 0.8rem;
  font-weight: 600;
  color: white;
}

.result-details h4 {
  font-size: 1.1rem;
  font-weight: 600;
  color: white;
  margin-bottom: 1rem;
}

.result-details ul {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.result-details li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  font-size: 0.9rem;
}

.result-details li span:first-child {
  color: rgba(255, 255, 255, 0.7);
}

.result-details li span:last-child {
  font-weight: 600;
  color: white;
}

.quality-good {
  color: #00ff88 !important;
}

.result-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}

.secondary-btn,
.primary-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 30px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.secondary-btn {
  background: rgba(255, 255, 255, 0.1);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.secondary-btn:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
}

.primary-btn {
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  color: white;
}

.primary-btn:hover {
  background: linear-gradient(135deg, #00f5ff 0%, #0044cc 100%);
  transform: translateY(-2px);
}

.secondary-btn svg,
.primary-btn svg {
  width: 16px;
  height: 16px;
}

@media (max-width: 768px) {
  .upload-container {
    padding: 1rem;
  }
  
  .upload-card {
    padding: 2rem;
  }
  
  .upload-area {
    padding: 2rem;
    min-height: 250px;
  }
  
  .action-buttons {
    flex-direction: column;
    align-items: center;
  }
  
  .browse-btn,
  .analyze-btn {
    width: 100%;
    max-width: 300px;
  }
  
  .result-actions {
    flex-direction: column;
    align-items: center;
  }
  
  .secondary-btn,
  .primary-btn {
    width: 100%;
    max-width: 300px;
    justify-content: center;
  }
}

@media (max-width: 480px) {
  .header-section {
    margin-bottom: 2rem;
  }
  
  .upload-area {
    padding: 1.5rem;
    min-height: 200px;
  }
  
  .upload-icon {
    width: 60px;
    height: 60px;
  }
  
  .particle {
    display: none;
  }
}

/* Similar Images Container - Adaptado al tema oscuro/claro */
.similar-images-container {
  margin-top: 2rem !important;
  padding: 2rem !important;
  background-color: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  border-radius: 12px !important;
  backdrop-filter: blur(10px) !important;
  width: 100% !important;
  box-sizing: border-box !important;
}

/* Tema claro */
@media (prefers-color-scheme: light) {
  .similar-images-container {
    background-color: rgba(0, 0, 0, 0.02) !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
  }
}

.similar-images-container h3 {
  color: rgba(255, 255, 255, 0.87) !important;
  font-size: 1.8rem !important;
  font-weight: 600 !important;
  margin-bottom: 1.5rem !important;
  text-align: center !important;
  font-family: system-ui, Avenir, Helvetica, Arial, sans-serif !important;
}

/* Tema claro para el título */
@media (prefers-color-scheme: light) {
  .similar-images-container h3 {
    color: #213547 !important;
  }
}

/* Grid 2x2 para las imágenes similares */
.similar-images-grid {
  display: grid !important;
  grid-template-columns: repeat(2, 1fr) !important;
  gap: 1.5rem !important;
  width: 100% !important;
  box-sizing: border-box !important;
}

/* Tarjetas de imágenes similares */
.similar-card {
  background-color: rgba(255, 255, 255, 0.08) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
  transition: all 0.3s ease !important;
  backdrop-filter: blur(5px) !important;
  position: relative !important;
  display: flex !important;
  flex-direction: column !important;
}

/* Tema claro para las tarjetas */
@media (prefers-color-scheme: light) {
  .similar-card {
    background-color: rgba(0, 0, 0, 0.02) !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
  }
}

.similar-card:hover {
  transform: translateY(-2px) !important;
  border-color: #646cff !important;
  box-shadow: 0 4px 12px rgba(100, 108, 255, 0.15) !important;
}

/* Contenedor para la imagen que mantiene el aspect ratio */
.similar-card .image-container {
  width: 100% !important;
  height: 180px !important;
  background-color: rgba(255, 255, 255, 0.02) !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  overflow: hidden !important;
}

@media (prefers-color-scheme: light) {
  .similar-card .image-container {
    background-color: rgba(0, 0, 0, 0.02) !important;
  }
}

.similar-card img {
  max-width: 100% !important;
  max-height: 100% !important;
  width: auto !important;
  height: auto !important;
  object-fit: contain !important;
  transition: transform 0.3s ease !important;
  display: block !important;
}

.similar-card:hover img {
  transform: scale(1.02) !important;
}

.similar-card-title {
  padding: 1rem 1rem 0.5rem !important;
  font-size: 1.1rem !important;
  font-weight: 600 !important;
  color: #646cff !important;
  text-align: center !important;
  font-family: system-ui, Avenir, Helvetica, Arial, sans-serif !important;
}

.similar-card-label {
  padding: 0 1rem 1rem !important;
  font-size: 0.9rem !important;
  color: rgba(255, 255, 255, 0.7) !important;
  text-align: center !important;
  line-height: 1.4 !important;
  font-family: system-ui, Avenir, Helvetica, Arial, sans-serif !important;
}

/* Tema claro para el label */
@media (prefers-color-scheme: light) {
  .similar-card-label {
    color: rgba(33, 53, 71, 0.7) !important;
  }
}

.similar-card-label strong {
  color: rgba(255, 255, 255, 0.87) !important;
  font-weight: 500 !important;
}

/* Tema claro para el strong */
@media (prefers-color-scheme: light) {
  .similar-card-label strong {
    color: #213547 !important;
  }
}

/* Responsive design */
@media (max-width: 768px) {
  .similar-images-container {
    padding: 1.5rem !important;
    margin-top: 1.5rem !important;
  }
  
  .similar-images-container h3 {
    font-size: 1.5rem !important;
    margin-bottom: 1rem !important;
  }
  
  .similar-images-grid {
    grid-template-columns: 1fr !important;
    gap: 1rem !important;
  }
  
  .similar-card .image-container {
    height: 160px !important;
  }
  
  .similar-card-title {
    font-size: 1rem !important;
    padding: 0.75rem 0.75rem 0.5rem !important;
  }
  
  .similar-card-label {
    font-size: 0.85rem !important;
    padding: 0 0.75rem 0.75rem !important;
  }
}

/* Animación suave de entrada */
.similar-card {
  animation: fadeInUp 0.5s ease-out !important;
}

.similar-card:nth-child(1) { animation-delay: 0.1s !important; }
.similar-card:nth-child(2) { animation-delay: 0.2s !important; }
.similar-card:nth-child(3) { animation-delay: 0.3s !important; }
.similar-card:nth-child(4) { animation-delay: 0.4s !important; }

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Estilo para cuando hay más de 4 imágenes */
.similar-images-grid .similar-card:nth-child(n+5) {
  display: none !important;
}

/* Mostrar indicador si hay más imágenes */
.similar-images-container::after {
  content: '';
  display: none;
}

.similar-images-container[data-total-images] .similar-images-grid::after {
  content: attr(data-remaining-count) ' more similar cases...';
  grid-column: 1 / -1;
  text-align: center;
  padding: 1rem;
  color: rgba(255, 255, 255, 0.6);
  font-style: italic;
  font-size: 0.9rem;
}

@media (prefers-color-scheme: light) {
  .similar-images-container[data-total-images] .similar-images-grid::after {
    color: rgba(33, 53, 71, 0.6);
  }
}
</style>
