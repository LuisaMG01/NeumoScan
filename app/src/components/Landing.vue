
<template>
  <div class="landing">
    <!-- Animated Background -->
    <div class="animated-background">
      <div class="floating-shapes">
        <div class="shape shape-1"></div>
        <div class="shape shape-2"></div>
        <div class="shape shape-3"></div>
        <div class="shape shape-4"></div>
        <div class="shape shape-5"></div>
      </div>
      <div class="grid-pattern"></div>
    </div>

    <!-- Hero Section -->
    <section class="hero-section">
      <div class="hero-container">
        <!-- Status Badge -->
        <div class="status-badge">
          <div class="pulse-dot"></div>
          <span>AI-Powered • Real-time Analysis</span>
        </div>

        <!-- Main Title -->
        <h1 class="hero-title">
          <span class="title-line">
            <span class="gradient-text">NeumoScan</span>
          </span>
          <span class="title-line">
            <span class="subtitle-text">Pneumonia Detection</span>
          </span>
          <span class="title-line">
            <span class="highlight-text">Revolutionized</span>
          </span>
        </h1>

        <!-- Description -->
        <p class="hero-description">
          Revolutionary AI technology that analyzes chest X-rays in seconds to detect
          normal, bacterial, or viral pneumonia with <span class="highlight">medical-grade accuracy</span>.
        </p>

        <!-- CTA Button -->
        <div class="cta-container">
          <router-link to="/upload" class="cta-button">
            <span class="button-bg"></span>
            <span class="button-content">
              <span class="button-text">Start Diagnosis</span>
              <svg class="button-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M7 17L17 7M17 7H7M17 7V17"/>
              </svg>
            </span>
          </router-link>
          
          <div class="cta-note">
            <svg class="note-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2L2 7v10c0 5.55 3.84 10 9 11 5.16-1 9-5.45 9-11V7l-10-5z"/>
              <path d="M9 12l2 2 4-4"/>
            </svg>
            No registration required • Results in seconds
          </div>
        </div>

        <!-- Stats -->
        <div class="stats-container">
          <div class="stat-item">
            <div class="stat-number">99.2%</div>
            <div class="stat-label">Accuracy Rate</div>
          </div>
          <div class="stat-divider"></div>
          <div class="stat-item">
            <div class="stat-number">&lt;30s</div>
            <div class="stat-label">Analysis Time</div>
          </div>
          <div class="stat-divider"></div>
          <div class="stat-item">
            <div class="stat-number">50k+</div>
            <div class="stat-label">Diagnoses Made</div>
          </div>
        </div>
      </div>

      <!-- Scroll Indicator -->
      <div class="scroll-indicator">
        <div class="scroll-line"></div>
        <div class="scroll-text">Scroll to explore</div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="features-section">
      <div class="features-container">
        <div class="section-header">
          <h2 class="section-title">Why Choose NeumoScan?</h2>
          <p class="section-subtitle">Advanced AI technology meets medical excellence</p>
        </div>

        <div class="features-grid">
          <div class="feature-card">
            <div class="feature-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9 12l2 2 4-4"/>
                <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"/>
                <path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"/>
                <path d="M21 12c0 1.66-1.34 3-3 3s-3-1.34-3-3"/>
                <path d="M3 12c0 1.66 1.34 3 3 3s3-1.34 3-3"/>
              </svg>
            </div>
            <h3>Medical Grade Accuracy</h3>
            <p>99.2% accuracy rate validated by radiologists and medical professionals worldwide</p>
          </div>

          <div class="feature-card">
            <div class="feature-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
              </svg>
            </div>
            <h3>Lightning Fast</h3>
            <p>Get results in under 30 seconds with our optimized neural network architecture</p>
          </div>

          <div class="feature-card">
            <div class="feature-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2L2 7v10c0 5.55 3.84 10 9 11 5.16-1 9-5.45 9-11V7l-10-5z"/>
              </svg>
            </div>
            <h3>Secure & Private</h3>
            <p>Your medical data is encrypted and never stored. Complete privacy guaranteed</p>
          </div>

          <div class="feature-card">
            <div class="feature-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
              </svg>
            </div>
            <h3>Multi-Type Detection</h3>
            <p>Distinguishes between normal, bacterial, and viral pneumonia with precision</p>
          </div>
        </div>
      </div>
    </section>

    <!-- How It Works Section -->
    <section class="how-it-works-section">
      <div class="how-it-works-container">
        <div class="section-header">
          <h2 class="section-title">How NeumoScan Works</h2>
          <p class="section-subtitle">Advanced AI technology simplified in three steps</p>
        </div>

        <div class="process-steps">
          <div class="step-card">
            <div class="step-number">01</div>
            <div class="step-content">
              <div class="step-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17,8 12,3 7,8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
              </div>
              <h3>Upload X-Ray</h3>
              <p>Simply upload your chest X-ray image in JPG or PNG format. Our system accepts standard radiological images.</p>
            </div>
          </div>

          <div class="step-card">
            <div class="step-number">02</div>
            <div class="step-content">
              <div class="step-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M12 2L2 7v10c0 5.55 3.84 10 9 11 5.16-1 9-5.45 9-11V7l-10-5z"/>
                  <path d="M9 12l2 2 4-4"/>
                </svg>
              </div>
              <h3>AI Analysis</h3>
              <p>Our neural network analyzes the image patterns, identifying key indicators of pneumonia and lung abnormalities.</p>
            </div>
          </div>

          <div class="step-card">
            <div class="step-number">03</div>
            <div class="step-content">
              <div class="step-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M9 11H5a2 2 0 0 0-2 2v3c0 1.1.9 2 2 2h4l3 3h4a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-4l-3-3z"/>
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
              </div>
              <h3>Get Results</h3>
              <p>Receive detailed analysis results including diagnosis classification and confidence levels in seconds.</p>
            </div>
          </div>
        </div>

        <div class="technology-info">
          <div class="tech-content">
            <h3>Powered by Deep Learning</h3>
            <p>
              NeumoScan uses advanced convolutional neural networks trained on medical imaging data 
              to identify patterns associated with different types of pneumonia. The system provides 
              educational insights for learning purposes.
            </p>
            <div class="tech-features">
              <div class="tech-feature">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M12 2L2 7v10c0 5.55 3.84 10 9 11 5.16-1 9-5.45 9-11V7l-10-5z"/>
                </svg>
                <span>Privacy First</span>
              </div>
              <div class="tech-feature">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                </svg>
                <span>Fast Processing</span>
              </div>
              <div class="tech-feature">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M9 12l2 2 4-4"/>
                </svg>
                <span>Educational Tool</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script>
export default {
  name: 'Landing',
  mounted() {
    // Add scroll animations
    this.initScrollAnimations();
  },
  methods: {
    initScrollAnimations() {
      // Simple scroll animation observer
      const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
      };

      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
          }
        });
      }, observerOptions);

      // Observe all feature cards
      document.querySelectorAll('.feature-card').forEach(card => {
        observer.observe(card);
      });
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

.landing {
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

.floating-shapes {
  position: absolute;
  width: 100%;
  height: 100%;
}

.shape {
  position: absolute;
  border-radius: 50%;
  opacity: 0.6;
  animation: float 8s ease-in-out infinite;
}

.shape-1 {
  width: 120px;
  height: 120px;
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  top: 10%;
  left: 10%;
  animation-delay: 0s;
}

.shape-2 {
  width: 80px;
  height: 80px;
  background: linear-gradient(135deg, #ff0080 0%, #ff4569 100%);
  top: 20%;
  right: 20%;
  animation-delay: -2s;
}

.shape-3 {
  width: 60px;
  height: 60px;
  background: linear-gradient(135deg, #00ff88 0%, #00cc69 100%);
  bottom: 30%;
  left: 30%;
  animation-delay: -4s;
}

.shape-4 {
  width: 100px;
  height: 100px;
  background: linear-gradient(135deg, #ffaa00 0%, #ff6600 100%);
  bottom: 10%;
  right: 10%;
  animation-delay: -6s;
}

.shape-5 {
  width: 140px;
  height: 140px;
  background: linear-gradient(135deg, #aa00ff 0%, #6600cc 100%);
  top: 60%;
  left: 60%;
  animation-delay: -1s;
}

@keyframes float {
  0%, 100% { 
    transform: translateY(0px) translateX(0px) rotate(0deg);
    opacity: 0.6;
  }
  25% { 
    transform: translateY(-20px) translateX(10px) rotate(90deg);
    opacity: 0.8;
  }
  50% { 
    transform: translateY(-10px) translateX(-10px) rotate(180deg);
    opacity: 0.4;
  }
  75% { 
    transform: translateY(10px) translateX(5px) rotate(270deg);
    opacity: 0.7;
  }
}

.grid-pattern {
  position: absolute;
  width: 100%;
  height: 100%;
  background-image: 
    linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
  background-size: 50px 50px;
  animation: gridMove 20s linear infinite;
}

@keyframes gridMove {
  0% { transform: translate(0, 0); }
  100% { transform: translate(50px, 50px); }
}

/* Hero Section */
.hero-section {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  position: relative;
  z-index: 2;
}

.hero-container {
  text-align: center;
  max-width: 1200px;
  margin: 0 auto;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 50px;
  margin-bottom: 3rem;
  backdrop-filter: blur(20px);
  font-size: 0.9rem;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.8);
}

.pulse-dot {
  width: 8px;
  height: 8px;
  background: #00ff88;
  border-radius: 50%;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { 
    opacity: 1;
    transform: scale(1);
  }
  50% { 
    opacity: 0.5;
    transform: scale(1.2);
  }
}

.hero-title {
  font-size: clamp(3rem, 8vw, 8rem);
  font-weight: 900;
  line-height: 0.9;
  margin-bottom: 2rem;
  letter-spacing: -0.02em;
}

.title-line {
  display: block;
  margin-bottom: 0.5rem;
}

.gradient-text {
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 50%, #0044cc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: gradientShift 3s ease-in-out infinite;
}

@keyframes gradientShift {
  0%, 100% { filter: hue-rotate(0deg); }
  50% { filter: hue-rotate(30deg); }
}

.subtitle-text {
  background: linear-gradient(135deg, #ffffff 0%, #cccccc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.highlight-text {
  background: linear-gradient(135deg, #ff0080 0%, #ff4569 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-description {
  font-size: 1.4rem;
  font-weight: 400;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 3rem;
  line-height: 1.6;
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
}

.hero-description .highlight {
  color: #00ff88;
  font-weight: 600;
}

.cta-container {
  margin-bottom: 4rem;
}

.cta-button {
  position: relative;
  display: inline-flex;
  align-items: center;
  padding: 1.5rem 3rem;
  border-radius: 50px;
  text-decoration: none;
  font-size: 1.2rem;
  font-weight: 600;
  color: white;
  margin-bottom: 2rem;
  transition: all 0.3s ease;
  overflow: hidden;
}

.button-bg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  border-radius: 50px;
  transition: all 0.3s ease;
}

.cta-button:hover .button-bg {
  background: linear-gradient(135deg, #00f5ff 0%, #0044cc 100%);
  transform: scale(1.05);
}

.button-content {
  position: relative;
  display: flex;
  align-items: center;
  gap: 1rem;
  z-index: 1;
}

.button-icon {
  width: 20px;
  height: 20px;
  transition: transform 0.3s ease;
}

.cta-button:hover .button-icon {
  transform: translateX(5px);
}

.cta-note {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.6);
  font-weight: 500;
}

.note-icon {
  width: 16px;
  height: 16px;
  color: #00ff88;
}

.stats-container {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 2rem;
  flex-wrap: wrap;
}

.stat-item {
  text-align: center;
}

.stat-number {
  font-size: 3rem;
  font-weight: 800;
  background: linear-gradient(135deg, #00ff88 0%, #00cc69 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.5rem;
}

.stat-label {
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.6);
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 500;
}

.stat-divider {
  width: 1px;
  height: 40px;
  background: rgba(255, 255, 255, 0.2);
}

.scroll-indicator {
  position: absolute;
  bottom: 2rem;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  animation: bounce 2s ease-in-out infinite;
}

.scroll-line {
  width: 2px;
  height: 30px;
  background: linear-gradient(to bottom, transparent, #00ff88);
}

.scroll-text {
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.5);
  text-transform: uppercase;
  letter-spacing: 1px;
}

@keyframes bounce {
  0%, 100% { transform: translateX(-50%) translateY(0); }
  50% { transform: translateX(-50%) translateY(-10px); }
}

/* Features Section */
.features-section {
  padding: 8rem 2rem;
  position: relative;
  z-index: 2;
  background: linear-gradient(180deg, transparent 0%, rgba(0, 0, 0, 0.5) 100%);
}

.features-container {
  max-width: 1200px;
  margin: 0 auto;
}

.section-header {
  text-align: center;
  margin-bottom: 5rem;
}

.section-title {
  font-size: 3.5rem;
  font-weight: 800;
  margin-bottom: 1rem;
  background: linear-gradient(135deg, #ffffff 0%, #cccccc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.section-subtitle {
  font-size: 1.3rem;
  color: rgba(255, 255, 255, 0.6);
  font-weight: 400;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
}

.feature-card {
  padding: 3rem;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  backdrop-filter: blur(20px);
  transition: all 0.3s ease;
  opacity: 0;
  transform: translateY(30px);
}

.feature-card:hover {
  background: rgba(255, 255, 255, 0.05);
  border-color: rgba(0, 245, 255, 0.3);
  transform: translateY(-10px);
}

.feature-icon {
  width: 60px;
  height: 60px;
  margin-bottom: 2rem;
  padding: 1rem;
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  border-radius: 15px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.feature-icon svg {
  width: 28px;
  height: 28px;
  color: white;
}

.feature-card h3 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  color: white;
}

.feature-card p {
  color: rgba(255, 255, 255, 0.7);
  line-height: 1.6;
  font-size: 1rem;
}

/* How It Works Section */
.how-it-works-section {
  padding: 8rem 2rem;
  background: rgba(0, 0, 0, 0.5);
  position: relative;
  z-index: 2;
}

.how-it-works-container {
  max-width: 1200px;
  margin: 0 auto;
}

.process-steps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 3rem;
  margin-bottom: 6rem;
}

.step-card {
  position: relative;
  padding: 2.5rem;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 24px;
  backdrop-filter: blur(20px);
  transition: all 0.3s ease;
  overflow: hidden;
}

.step-card:hover {
  background: rgba(255, 255, 255, 0.05);
  border-color: rgba(0, 245, 255, 0.3);
  transform: translateY(-5px);
}

.step-number {
  position: absolute;
  top: -1rem;
  right: 1.5rem;
  font-size: 4rem;
  font-weight: 900;
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  opacity: 0.3;
}

.step-content {
  position: relative;
  z-index: 1;
}

.step-icon {
  width: 60px;
  height: 60px;
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.step-icon svg {
  width: 28px;
  height: 28px;
  color: white;
}

.step-card h3 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  color: white;
}

.step-card p {
  color: rgba(255, 255, 255, 0.7);
  line-height: 1.6;
  font-size: 1rem;
}

.technology-info {
  text-align: center;
  padding: 3rem;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 24px;
  backdrop-filter: blur(20px);
}

.tech-content h3 {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
  background: linear-gradient(135deg, #00ff88 0%, #00cc69 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.tech-content p {
  font-size: 1.1rem;
  color: rgba(255, 255, 255, 0.7);
  line-height: 1.6;
  margin-bottom: 2rem;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
}

.tech-features {
  display: flex;
  justify-content: center;
  gap: 2rem;
  flex-wrap: wrap;
}

.tech-feature {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 50px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  transition: all 0.3s ease;
}

.tech-feature:hover {
  background: rgba(255, 255, 255, 0.1);
  border-color: rgba(0, 245, 255, 0.3);
}

.tech-feature svg {
  width: 18px;
  height: 18px;
  color: #00ff88;
}

.tech-feature span {
  font-size: 0.9rem;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.8);
}

/* Responsive Design */
@media (max-width: 768px) {
  .hero-title {
    font-size: 3rem;
  }
  
  .hero-description {
    font-size: 1.1rem;
  }
  
  .stats-container {
    gap: 1.5rem;
  }
  
  .stat-divider {
    display: none;
  }
  
  .features-grid {
    grid-template-columns: 1fr;
  }
  
  .feature-card {
    padding: 2rem;
  }
  
  .process-steps {
    grid-template-columns: 1fr;
    gap: 2rem;
  }
  
  .step-card {
    padding: 2rem;
  }
  
  .step-number {
    font-size: 3rem;
    top: -0.5rem;
  }
  
  .technology-info {
    padding: 2rem;
  }
  
  .tech-features {
    gap: 1rem;
  }
  
  .section-title {
    font-size: 2.5rem;
  }
}

@media (max-width: 480px) {
  .hero-section {
    padding: 1rem;
  }
  
  .features-section,
  .how-it-works-section {
    padding: 4rem 1rem;
  }
  
  .step-card {
    padding: 1.5rem;
  }
  
  .step-number {
    font-size: 2.5rem;
    top: -0.25rem;
  }
  
  .technology-info {
    padding: 1.5rem;
  }
  
  .tech-content h3 {
    font-size: 1.5rem;
  }
  
  .tech-features {
    flex-direction: column;
    align-items: center;
  }
  
  .tech-feature {
    width: 100%;
    max-width: 250px;
    justify-content: center;
  }
  
  .cta-button {
    padding: 1.2rem 2rem;
    font-size: 1rem;
  }
  
  .shape {
    display: none;
  }
}
</style>