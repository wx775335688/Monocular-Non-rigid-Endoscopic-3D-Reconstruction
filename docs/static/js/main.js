
(function () {
  async function videoExists(url) {
    try {
      const response = await fetch(url, { method: 'HEAD', cache: 'no-store' });
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  document.querySelectorAll('.video-block[data-video]').forEach(async (slot) => {
    const src = slot.getAttribute('data-video');
    const poster = slot.getAttribute('data-poster') || '';
    if (!src) return;
    const exists = await videoExists(src);
    if (!exists) return;
    slot.innerHTML = '';
    const video = document.createElement('video');
    video.controls = true;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = 'metadata';
    if (poster) video.poster = poster;
    const source = document.createElement('source');
    source.src = src;
    source.type = 'video/mp4';
    video.appendChild(source);
    slot.appendChild(video);
  });
})();
