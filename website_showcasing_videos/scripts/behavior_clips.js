

function scrollToTop() {
  //  window.scrollTo({ top: 0, behavior: 'smooth' });
  const targetPosition = 0;
  const startPosition = window.scrollY;
  const distance = targetPosition - startPosition;
  const duration = 500; // Duration of the scroll in ms
  let startTime = null;
  // Custom smooth scroll function
  function smoothScroll(currentTime)
  {
    if (!startTime) startTime = currentTime;
    const timeElapsed = currentTime - startTime;
    const progress = Math.min(timeElapsed / duration, 1); // Progress as a ratio
    const scrollAmount = startPosition + distance * progress;
    window.scrollTo(0, scrollAmount);
    if (timeElapsed < duration)
      requestAnimationFrame(smoothScroll); // Keep scrolling until duration is met
  }
  requestAnimationFrame(smoothScroll); // Start the animation
}

function jumpToVideo(video_element_id) {
  const element = document.getElementById(video_element_id);
  if (!element) return;

  const targetPosition = element.getBoundingClientRect().top + window.scrollY;
  const startPosition = window.scrollY;
  const distance = targetPosition - startPosition;
  const duration = 500; // Duration of the scroll in ms
  let startTime = null;
  // Custom smooth scroll function
  function smoothScroll(currentTime)
  {
    if (!startTime) startTime = currentTime;
    const timeElapsed = currentTime - startTime;
    const progress = Math.min(timeElapsed / duration, 1); // Progress as a ratio
    const scrollAmount = startPosition + distance * progress;
    window.scrollTo(0, scrollAmount);
    if (timeElapsed < duration)
      requestAnimationFrame(smoothScroll); // Keep scrolling until duration is met
  }
  requestAnimationFrame(smoothScroll); // Start the animation
}

// Remove the loading overlay when all video thumbnails have loaded
window.addEventListener('DOMContentLoaded', () => {
  const videos = document.querySelectorAll('video[poster]');
  const overlay = document.getElementById('loadingOverlay');
  let loadedCount = 0;

  if (videos.length === 0) {
    overlay.classList.add('hidden');
    return;
  }

  videos.forEach(video => {
    const posterUrl = video.getAttribute('poster');
    const img = new Image();

    img.onload = img.onerror = () => {
      loadedCount++;
      if (loadedCount === videos.length) {
        overlay.classList.add('hidden');
      }
    };

    img.src = posterUrl;
  });
});

// Match caption widths to video widths.
function matchCaptionWidths() {
  document.querySelectorAll('.video-caption').forEach(caption => {
    const video = caption.previousElementSibling;
    if (video && video.tagName.toLowerCase() === 'video') {
      caption.style.width = video.offsetWidth + 'px';
    }
  });
}

// Run on page load
window.addEventListener('load', matchCaptionWidths);

// Run again if window resizes (responsive videos)
window.addEventListener('resize', matchCaptionWidths);



