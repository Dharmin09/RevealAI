// home-highlight.js
// Show the highlight message for 3 seconds, then fade out
// Only show the highlight if on the home page (index.html or /)
window.addEventListener('DOMContentLoaded', function() {
  var msg = document.getElementById('home-highlight-msg');
  // Only show if on index.html or root
  var isHome = window.location.pathname.endsWith('index.html') || window.location.pathname === '/' || window.location.pathname === '';
  if (msg && isHome) {
    msg.style.opacity = 1;
    setTimeout(function() {
      msg.style.transition = 'opacity 1s';
      msg.style.opacity = 0;
      setTimeout(function() { msg.style.display = 'none'; }, 1000);
    }, 3000);
  } else if (msg) {
    msg.style.display = 'none';
  }
});
