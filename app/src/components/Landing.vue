
<template>
    <div class="landing">
      <!-- Hero Section -->
      <div class="hero-section">
        <div class="hero-background">
          <div class="gradient-orb orb-1"></div>
          <div class="gradient-orb orb-2"></div>
          <div class="gradient-orb orb-3"></div>
        </div>
        
        <div class="hero-content">
            <h1 class="hero-title">
            <span class="gradient-text">NeumoScan</span>
            <br>
            <span class="subtitle">Pneumonia Detection System</span>
          </h1>

            <div class="cta-content">
            <h2>Ready to Transform Pneumonia Diagnosis?</h2>
            <p>Join thousands of healthcare professionals using NeumoScan</p>
            
            <router-link to="/upload" class="cta-button">
                <span class="button-text">Start Diagnosis</span>
                <span class="button-icon">→</span>
            </router-link>
            
            <div class="cta-note">
                <span>✨ No registration required • Results in seconds</span>
            </div>

          
          
          <p class="hero-description">
            Revolutionary AI technology that analyzes chest X-rays in seconds to detect
            normal, bacterial, or viral pneumonia with medical-grade accuracy.
          </p>
          
          <div class="hero-stats">
            <div class="stat">
              <span class="stat-number">99.2%</span>
              <span class="stat-label">Accuracy</span>
            </div>
            <div class="stat">
              <span class="stat-number">&lt;30s</span>
              <span class="stat-label">Analysis Time</span>
            </div>
            <div class="stat">
              <span class="stat-number">50k+</span>
              <span class="stat-label">Diagnoses</span>
            </div>
          </div>
        </div>
      </div>

      </div>
    </div>
  </template>
  
  <script>
  
  </script>
  
  <style scoped>
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  
    .cta-section {
    background-image: url('./assets/xray.svg'); /* si está en public */
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    padding: 60px 20px;
    color: white; /* para que el texto contraste */
    position: relative;
    }

  .landing {
    min-height: 100vh;
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
    color: white;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    overflow-x: hidden;
    width: 100%;
    margin: 0;
    padding: 0;
  }
  
  /* Hero Section */
  .hero-section {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    padding: 2rem;
    width: 100%;
  }
  
  .hero-background {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    z-index: 1;
  }
  
  .gradient-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    animation: float 6s ease-in-out infinite;
  }
  
  .orb-1 {
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(77, 182, 172, 0.3) 0%, transparent 70%);
    top: 10%;
    left: 10%;
    animation-delay: 0s;
  }
  
  .orb-2 {
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(255, 68, 68, 0.2) 0%, transparent 70%);
    top: 60%;
    right: 10%;
    animation-delay: -2s;
  }
  
  .orb-3 {
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(255, 184, 68, 0.2) 0%, transparent 70%);
    bottom: 20%;
    left: 30%;
    animation-delay: -4s;
  }
  
  @keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(180deg); }
  }
  
  .hero-content {
    position: relative;
    z-index: 2;
    text-align: center;
    width: 100%;
  }
  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(77, 182, 172, 0.1);
    border: 1px solid rgba(77, 182, 172, 0.3);
    border-radius: 50px;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
  }
  
  .badge-icon {
    font-size: 1.2rem;
  }
  
  .badge-text {
    font-size: 0.9rem;
    color: #4db6ac;
    font-weight: 500;
  }
  
  .hero-title {
    font-size: clamp(3rem, 8vw, 6rem);
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 1.5rem;
  }
  
  .gradient-text {
    background: linear-gradient(135deg, #4db6ac 0%, #26a69a 50%, #00796b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .subtitle {
    font-size: clamp(1.5rem, 4vw, 2.5rem);
    font-weight: 300;
    color: #b0b0b0;
  }
  
  .hero-description {
    font-size: 1.3rem;
    color: #888;
    margin-bottom: 3rem;
    line-height: 1.6;
  }
  
  .hero-stats {
    display: flex;
    justify-content: center;
    gap: 3rem;
    flex-wrap: wrap;
  }
  
  .stat {
    text-align: center;
  }
  
  .stat-number {
    display: block;
    font-size: 2.5rem;
    font-weight: 700;
    color: #4db6ac;
    margin-bottom: 0.5rem;
  }
  
  .stat-label {
    font-size: 0.9rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  
  /* Carousel Section */
  .carousel-section {
    padding: 8rem 2rem;
    background: linear-gradient(180deg, #0f0f0f 0%, #1a1a1a 100%);
  }
  
  .section-header {
    text-align: center;
    margin-bottom: 4rem;
  }
  
  .section-header h2 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #fff 0%, #ccc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .section-header p {
    font-size: 1.2rem;
    color: #888;
  }
  
  .carousel-container {
    width: 100%;
    margin: 0 auto;
    padding: 0 2rem;
  }
  
  .carousel-wrapper {
    position: relative;
    height: 500px;
    margin-bottom: 2rem;
  }
  
  .carousel-slide {
    position: absolute;
    width: 100%;
    height: 100%;
    opacity: 0;
    transform: translateX(100px);
    transition: all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    cursor: pointer;
  }
  
  .carousel-slide.active {
    opacity: 1;
    transform: translateX(0);
  }
  
  .slide-content {
    position: relative;
    width: 100%;
    height: 100%;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    background: #2a2a2a;
  }
  
  .slide-content img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  .slide-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, rgba(0, 0, 0, 0.8));
    padding: 2rem;
    color: white;
  }
  
  .slide-info h3 {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }
  
  .slide-info p {
    color: #ccc;
    margin-bottom: 1rem;
  }
  
  .diagnosis-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  
  .diagnosis-badge.normal {
    background: rgba(77, 182, 172, 0.2);
    color: #4db6ac;
    border: 1px solid #4db6ac;
  }
  
  .diagnosis-badge.bacterial {
    background: rgba(255, 68, 68, 0.2);
    color: #ff4444;
    border: 1px solid #ff4444;
  }
  
  .diagnosis-badge.viral {
    background: rgba(255, 184, 68, 0.2);
    color: #ffb844;
    border: 1px solid #ffb844;
  }
  
  .carousel-indicators {
    display: flex;
    justify-content: center;
    gap: 1rem;
  }
  
  .indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: none;
    background: rgba(255, 255, 255, 0.3);
    cursor: pointer;
    transition: all 0.3s ease;
  }
  
  .indicator.active {
    background: #4db6ac;
    transform: scale(1.2);
  }
  
  /* Features Section */
  .features-section {
    padding: 8rem 2rem;
    background: #0f0f0f;
  }
  
  .features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    width: 100%;
    margin: 0 auto;
    padding: 0 2rem;
  }
  
  .feature-card {
    background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
    padding: 2.5rem;
    border-radius: 20px;
    text-align: center;
    border: 1px solid rgba(77, 182, 172, 0.1);
    transition: all 0.3s ease;
    cursor: pointer;
  }
  
  .feature-card:hover {
    transform: translateY(-10px);
    border-color: rgba(77, 182, 172, 0.3);
    box-shadow: 0 20px 40px rgba(77, 182, 172, 0.1);
  }
  
  .feature-icon {
    font-size: 3rem;
    margin-bottom: 1.5rem;
  }
  
  .feature-card h3 {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #fff;
  }
  
  .feature-card p {
    color: #888;
    line-height: 1.6;
  }
   
  .cta-content {
    text-align: center;
    width: 100%;
    margin: 0 auto;
    padding: 0 2rem;
  }
  
  .cta-content h2 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #fff 0%, #ccc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .cta-content p {
    font-size: 1.2rem;
    color: #888;
    margin-bottom: 3rem;
  }
  
  .cta-button {
    display: inline-flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem 3rem;
    background: linear-gradient(135deg, #4db6ac 0%, #26a69a 100%);
    color: white;
    text-decoration: none;
    border-radius: 50px;
    font-size: 1.2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 10px 30px rgba(77, 182, 172, 0.3);
  }
  
  .cta-button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(77, 182, 172, 0.4);
  }
  
  .button-icon {
    font-size: 1.5rem;
    transition: transform 0.3s ease;
  }
  
  .cta-button:hover .button-icon {
    transform: translateX(5px);
  }
  
  .cta-note {
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #666;
  }
  
  /* Responsive Design */
  @media (max-width: 768px) {
    .hero-stats {
      gap: 2rem;
    }
    
    .features-grid {
      grid-template-columns: 1fr;
    }
    
    .carousel-wrapper {
      height: 400px;
    }
    
    .slide-overlay {
      padding: 1.5rem;
    }
    
    .cta-content h2 {
      font-size: 2rem;
    }
  }
  
  @media (max-width: 480px) {
    .hero-section {
      padding: 1rem;
    }
    
    .carousel-section,
    .features-section,
    .cta-section {
      padding: 4rem 1rem;
    }
    
    .feature-card {
      padding: 2rem;
    }
  }
  </style>