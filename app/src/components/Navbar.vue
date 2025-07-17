<template>
  <nav class="navbar" :class="{ 'scrolled': isScrolled }">
    <div class="navbar-container">
      <!-- Logo Section -->
      <router-link to="/" class="logo">
        <div class="logo-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
          </svg>
        </div>
        <span class="logo-text">
          <span class="logo-gradient">NeumoScan</span>
        </span>
      </router-link>

      <!-- Navigation Links -->
      <ul class="nav-links">
        <li>
          <router-link to="/" class="nav-link">
            <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
              <polyline points="9,22 9,12 15,12 15,22"/>
            </svg>
            <span class="nav-text">Home</span>
          </router-link>
        </li>
        <li>
          <router-link to="/upload" class="nav-link upload-link">
            <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17,8 12,3 7,8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <span class="nav-text">Upload Image</span>
          </router-link>
        </li>
      </ul>

      <!-- Mobile Menu Toggle -->
      <button class="mobile-toggle" @click="toggleMobileMenu" :class="{ 'active': mobileMenuOpen }">
        <span class="hamburger-line"></span>
        <span class="hamburger-line"></span>
        <span class="hamburger-line"></span>
      </button>
    </div>

    <!-- Mobile Menu -->
    <div class="mobile-menu" :class="{ 'active': mobileMenuOpen }">
      <div class="mobile-nav-links">
        <router-link to="/" class="mobile-nav-link" @click="closeMobileMenu">
          <svg class="mobile-nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
            <polyline points="9,22 9,12 15,12 15,22"/>
          </svg>
          <span>Home</span>
        </router-link>
        <router-link to="/upload" class="mobile-nav-link" @click="closeMobileMenu">
          <svg class="mobile-nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17,8 12,3 7,8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          <span>Upload Image</span>
        </router-link>
      </div>
    </div>
  </nav>
</template>

<script>
export default {
  name: 'Navbar',
  data() {
    return {
      isScrolled: false,
      mobileMenuOpen: false,
    };
  },
  mounted() {
    window.addEventListener('scroll', this.handleScroll);
  },
  beforeDestroy() {
    window.removeEventListener('scroll', this.handleScroll);
  },
  methods: {
    handleScroll() {
      this.isScrolled = window.scrollY > 50;
    },
    toggleMobileMenu() {
      this.mobileMenuOpen = !this.mobileMenuOpen;
    },
    closeMobileMenu() {
      this.mobileMenuOpen = false;
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

.navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  background: rgba(10, 10, 10, 0.80);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  transition: all 0.3s ease;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.navbar.scrolled {
  background: rgba(10, 10, 10, 0.95);
  backdrop-filter: blur(25px);
  -webkit-backdrop-filter: blur(25px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.navbar-container {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
}

/* Logo Section */
.logo {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  text-decoration: none;
  transition: all 0.3s ease;
}

.logo:hover {
  transform: translateY(-2px);
}

.logo-icon {
  width: 40px;
  height: 40px;
  padding: 0.5rem;
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
}

.logo:hover .logo-icon {
  background: linear-gradient(135deg, #00f5ff 0%, #0044cc 100%);
  transform: rotate(5deg);
}

.logo-icon svg {
  width: 20px;
  height: 20px;
  color: white;
}

.logo-text {
  font-size: 1.5rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}

.logo-gradient {
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: logoGlow 3s ease-in-out infinite;
}

@keyframes logoGlow {
  0%, 100% { filter: hue-rotate(0deg); }
  50% { filter: hue-rotate(30deg); }
}

/* Navigation Links */
.nav-links {
  display: flex;
  list-style: none;
  gap: 0.5rem;
  align-items: center;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  text-decoration: none;
  color: rgba(255, 255, 255, 0.8);
  font-weight: 500;
  font-size: 0.95rem;
  border-radius: 30px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.nav-link::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 30px;
  opacity: 0;
  transition: opacity 0.3s ease;
}

.nav-link:hover::before {
  opacity: 1;
}

.nav-link:hover {
  color: white;
  transform: translateY(-2px);
}

.nav-link.router-link-exact-active {
  color: white;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.nav-link.router-link-exact-active::before {
  opacity: 0;
}

.nav-icon {
  width: 18px;
  height: 18px;
  transition: transform 0.3s ease;
}

.nav-link:hover .nav-icon {
  transform: scale(1.1);
}

.nav-text {
  white-space: nowrap;
}

/* Upload Link Special Styling */
.upload-link {
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  color: white;
  border: 1px solid rgba(0, 245, 255, 0.3);
  font-weight: 600;
}

.upload-link::before {
  background: rgba(255, 255, 255, 0.2);
}

.upload-link:hover {
  background: linear-gradient(135deg, #00f5ff 0%, #0044cc 100%);
  transform: translateY(-3px);
  box-shadow: 0 10px 25px rgba(0, 245, 255, 0.3);
}

.upload-link.router-link-exact-active {
  background: linear-gradient(135deg, #00f5ff 0%, #0044cc 100%);
  border-color: rgba(0, 245, 255, 0.5);
}

/* Mobile Toggle */
.mobile-toggle {
  display: none;
  flex-direction: column;
  gap: 4px;
  background: none;
  border: none;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.mobile-toggle:hover {
  background: rgba(255, 255, 255, 0.1);
}

.hamburger-line {
  width: 24px;
  height: 2px;
  background: white;
  border-radius: 1px;
  transition: all 0.3s ease;
}

.mobile-toggle.active .hamburger-line:nth-child(1) {
  transform: rotate(45deg) translate(7px, 7px);
}

.mobile-toggle.active .hamburger-line:nth-child(2) {
  opacity: 0;
}

.mobile-toggle.active .hamburger-line:nth-child(3) {
  transform: rotate(-45deg) translate(7px, -7px);
}

/* Mobile Menu */
.mobile-menu {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: rgba(10, 10, 10, 0.95);
  backdrop-filter: blur(25px);
  -webkit-backdrop-filter: blur(25px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  transform: translateY(-100%);
  opacity: 0;
  visibility: hidden;
  transition: all 0.3s ease;
}

.mobile-menu.active {
  transform: translateY(0);
  opacity: 1;
  visibility: visible;
}

.mobile-nav-links {
  display: flex;
  flex-direction: column;
  padding: 1rem;
  gap: 0.5rem;
}

.mobile-nav-link {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  text-decoration: none;
  color: rgba(255, 255, 255, 0.8);
  font-weight: 500;
  border-radius: 12px;
  transition: all 0.3s ease;
}

.mobile-nav-link:hover {
  background: rgba(255, 255, 255, 0.1);
  color: white;
}

.mobile-nav-link.router-link-exact-active {
  background: rgba(0, 245, 255, 0.1);
  color: #00f5ff;
  border: 1px solid rgba(0, 245, 255, 0.3);
}

.mobile-nav-icon {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

/* Responsive Design */
@media (max-width: 768px) {
  .navbar-container {
    padding: 0.75rem 1rem;
  }
  
  .nav-links {
    display: none;
  }
  
  .mobile-toggle {
    display: flex;
  }
  
  .logo-text {
    font-size: 1.3rem;
  }
  
  .logo-icon {
    width: 36px;
    height: 36px;
  }
  
  .logo-icon svg {
    width: 18px;
    height: 18px;
  }
}

@media (max-width: 480px) {
  .navbar-container {
    padding: 0.5rem 1rem;
  }
  
  .logo-text {
    font-size: 1.2rem;
  }
  
  .logo-icon {
    width: 32px;
    height: 32px;
  }
  
  .logo-icon svg {
    width: 16px;
    height: 16px;
  }
}

/* Smooth transitions for route changes */
.nav-link,
.mobile-nav-link {
  position: relative;
}

.nav-link::after,
.mobile-nav-link::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 50%;
  width: 0;
  height: 2px;
  background: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
  transition: all 0.3s ease;
  transform: translateX(-50%);
}

.nav-link.router-link-exact-active::after,
.mobile-nav-link.router-link-exact-active::after {
  width: 60%;
}

/* Add subtle animation on page load */
.navbar-container {
  animation: slideDown 0.5s ease-out;
}

@keyframes slideDown {
  from {
    transform: translateY(-100%);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

/* Enhance focus states for accessibility */
.nav-link:focus,
.mobile-nav-link:focus,
.logo:focus,
.mobile-toggle:focus {
  outline: 2px solid #00f5ff;
  outline-offset: 2px;
}

/* Add subtle glow effect on active states */
.nav-link.router-link-exact-active,
.mobile-nav-link.router-link-exact-active {
  box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
}
</style>
  