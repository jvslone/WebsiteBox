// ... (The rest of this file is unchanged) ...
(() => {
const canvas = document.getElementById('flow_canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
console.log('Canvas initialized:', canvas, ctx);
})();