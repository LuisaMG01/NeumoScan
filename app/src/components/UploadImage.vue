<template>
  <div class="upload">
    <h2>Upload Chest X-Ray</h2>
    <p>Select a JPG or PNG image. Our AI will analyze it for pneumonia signs.</p>

    <label for="fileInput" class="file-label">📁 Choose Image</label>
    <input id="fileInput" type="file" @change="onFileChange" accept="image/*" />

    <button @click="predict" :disabled="!selectedFile">🔍 Analyze</button>

    <div v-if="selectedFile" class="preview-section">
      <div class="image-preview">
        <img :src="imagePreviewUrl" alt="Preview" />
      </div>

      <div v-if="prediction" class="result">
        <h3>Result</h3>
        <p><span>Class:</span> {{ prediction.class }}</p>
        <p><span>Confidence:</span> {{ (prediction.confidence * 100).toFixed(2) }}%</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      selectedFile: null,
      prediction: null,
      imagePreviewUrl: null,
    };
  },
  methods: {
    onFileChange(event) {
      this.selectedFile = event.target.files[0];
      this.prediction = null;

      if (this.selectedFile) {
        this.imagePreviewUrl = URL.createObjectURL(this.selectedFile);
      } else {
        this.imagePreviewUrl = null;
      }
    },
    async predict() {
      const formData = new FormData();
      formData.append("file", this.selectedFile);

      try {
        const response = await fetch("http://localhost:8000/predict", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        this.prediction = data;
      } catch (error) {
        console.error("Prediction error:", error);
      }
    }
  }
};
</script>

<style scoped>
.upload {
  max-width: 600px;
  margin: 5rem auto;
  padding: 2rem;
  border-radius: 1rem;
  background: #fafafa;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
  text-align: center;
  font-family: 'Inter', sans-serif;
}

h2 {
  font-size: 1.8rem;
  font-weight: 600;
  color: #2b2d42;
  margin-bottom: 1rem;
}

p {
  font-size: 0.95rem;
  color: #555;
  margin-bottom: 1.5rem;
}

input[type="file"] {
  display: none;
}

.file-label {
  background-color: #00bcd4;
  color: white;
  padding: 0.6rem 1.4rem;
  border-radius: 2rem;
  cursor: pointer;
  font-weight: 500;
  transition: background 0.3s ease;
}

.file-label:hover {
  background-color: #0097a7;
}

button {
  margin-top: 1rem;
  background-color: #2b2d42;
  color: white;
  border: none;
  padding: 0.6rem 1.6rem;
  border-radius: 2rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.3s ease;
}

button:hover {
  background-color: #1a1c30;
}

button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.preview-section {
  margin-top: 2rem;
  display: flex;
  flex-wrap: wrap;
  gap: 2rem;
  justify-content: center;
}

.image-preview {
  width: 280px;
  height: auto;
  border-radius: 0.75rem;
  overflow: hidden;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
  background: #f0f0f0;
}

.image-preview img {
  width: 100%;
  height: auto;
  object-fit: cover;
}

.result {
  width: 280px;
  background-color: #e0f7fa;
  border-radius: 0.75rem;
  padding: 1.2rem;
  text-align: left;
  font-size: 0.95rem;
  color: #00363a;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
}

.result h3 {
  font-size: 1.1rem;
  margin-bottom: 0.8rem;
  color: #006064;
}

.result span {
  font-weight: 600;
}
</style>
