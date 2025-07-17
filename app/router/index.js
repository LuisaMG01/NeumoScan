import { createRouter, createWebHistory } from 'vue-router';
import Landing from '../src/components/Landing.vue';
import UploadImage from '../src/components/UploadImage.vue';

const routes = [
  { path: '/', component: Landing },
  { path: '/upload', component: UploadImage },
];

export default createRouter({
  history: createWebHistory(),
  routes,
});