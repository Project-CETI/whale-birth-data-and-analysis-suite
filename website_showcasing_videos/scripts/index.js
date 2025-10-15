
// Track current category globally
let currentCategory = 'raw';

function showCategory(category) {
  currentCategory = category;

  // Show/hide the appropriate video group
  ['raw', 'masks', 'boxes', 'masks-composite', 'boxes-composite'].forEach(group => {
    document.getElementById(group).classList.toggle('hidden', group !== category);
  });

  // Reset all category buttons
  document.querySelectorAll('.button-row button').forEach(btn => {
    btn.classList.remove('active');
  });

  // Set active class on the selected button
  const activeButton = document.getElementById(`button-show-${category}`);
  if (activeButton) {
    activeButton.classList.add('active');
  }
}

function jumpToVideo(name) {
  const category = currentCategory;

  const videoId = `${category}-${name}`;
  const element = document.getElementById(videoId);
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


// Show default on page load
window.onload = () => showCategory('masks-composite');

// Handle clicks on the timeline image.
const image = document.getElementById("bannerImage");
const x_starts = [160.0947435736656, 192.62951857339255, 225.00192538228904, 379.36223447702145, 406.8142881566828, 412.7480290873484, 632.7135472622784, 665.2529040553353, 697.8019972573627, 703.6542676985264, 724.586447255178, 757.1899495303632, 789.8031881462445, 891.2880640252071, 923.8275640238415, 956.3717890213836, 1237.2613938212394];
const x_ends = [192.6294278708371, 219.50775265802037, 241.55102312239734, 408.6003912708976, 412.5517268072475, 436.5731972694397, 665.2481790564277, 697.7972722584551, 700.8786881544373, 723.4283927012573, 757.1853677028959, 789.7984631473367, 807.1701404165137, 923.8226958193562, 956.367064022476, 988.9064208496701, 1269.3095558158557];
const y_starts = [18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0];
const y_ends = [310.0, 310.0, 310.0, 310.0, 310.0, 310.0, 310.0, 310.0, 310.0, 310.0, 310.0, 310.0, 310.0, 310.0, 310.0, 310.0, 310.0];
const imageMap_video_keys = ["1688827433752","1688827660979","1688827887072","1688828965144","1688829151574","1688829193016","1688830733272","1688830960531","1688831187858","1688831228731","1688831374924","1688831602631","1688831830406","1688832540499","1688832767759","1688832995052","1688841618482"];
image.addEventListener("click", (event) => {
  const rect = image.getBoundingClientRect();

  // Convert click position to image-relative coordinates
  const clickX = event.clientX - rect.left;
  const clickY = event.clientY - rect.top;

  // Scale to native image resolution
  const scaleX = image.naturalWidth / image.clientWidth;
  const scaleY = image.naturalHeight / image.clientHeight;

  const nativeX = clickX * scaleX;
  const nativeY = clickY * scaleY;

//  console.log(`Clicked at native coords: (${nativeX.toFixed(1)}, ${nativeY.toFixed(1)})`);

  // Check which region the click falls into
  for (let i = 0; i < x_starts.length; i++) {
    if (
      nativeX >= x_starts[i] && nativeX <= x_ends[i] &&
      nativeY >= y_starts[i] && nativeY <= y_ends[i]
    ) {
      console.log(`Clicked inside region ${i} ` + imageMap_video_keys[i]);
       jumpToVideo(imageMap_video_keys[i]);
      return;
    }
  }

  console.log("Clicked outside of defined regions.");
});
// Add a listener to make the cursor a pointer over the clickable regions.
image.addEventListener("mousemove", (e) => {
  const rect = image.getBoundingClientRect();
  const scaleX = image.naturalWidth / image.clientWidth;
  const scaleY = image.naturalHeight / image.clientHeight;

  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  let inside = false;
  for (let i = 0; i < x_starts.length; i++) {
    if (
      x >= x_starts[i] &&
      x <= x_ends[i] &&
      y >= y_starts[i] &&
      y <= y_ends[i]
    ) {
      inside = true;
      break;
    }
  }

  image.style.cursor = inside ? "pointer" : "default";
});




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


