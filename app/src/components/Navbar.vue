<template>
    <nav class="navbar">
      <router-link to="/" class="logo">NeumoScan</router-link>
      <ul>
        <li><router-link to="/">Home</router-link></li>
        <li><router-link to="/upload">Upload Image</router-link></li>
      </ul>
    </nav>
  </template>
  
  <style scoped>
  .navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #00796b;
    padding: 1rem 2rem;
    color: white;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  
  .logo {
    font-weight: bold;
    font-size: 1.4rem;
    color: white;
    text-decoration: none;
  }
  
  ul {
    display: flex;
    list-style: none;
    gap: 1.5rem;
    margin: 0;
  }
  
  a {
    color: white;
    text-decoration: none;
  }
  
  a.router-link-exact-active {
    border-bottom: 2px solid white;
  }
  </style>
  