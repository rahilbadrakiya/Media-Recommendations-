// ═══════════════════════════════════════════
//  CineMate Ultra — Main Application v2
// ═══════════════════════════════════════════
import {
  auth, firebaseRegister, firebaseLogin, firebaseLoginGoogle,
  firebaseForgotPassword, firebaseLogout, getIdToken, onAuthChange
} from './firebase.js';

const API = 'http://localhost:8000/api';

// ── STATE ─────────────────────────────────────
let user = JSON.parse(localStorage.getItem('cm_user')) || null;
let currentMood = null;
let modalMovieId = null;
let cinemaMode = localStorage.getItem('cm_mode') || 'hollywood'; // 'hollywood' | 'indian'
let industry = localStorage.getItem('cm_industry') || 'all_indian'; // bollywood/tollywood/etc

const MOODS = [
  { id: 'happy', label: 'Happy 😄' },
  { id: 'excited', label: 'Excited 🔥' },
  { id: 'romantic', label: 'Romantic 💕' },
  { id: 'sad', label: 'Sad 😢' },
  { id: 'scared', label: 'Thrilled 😱' },
  { id: 'motivated', label: 'Motivated 💪' },
  { id: 'mindblown', label: 'Mind Blown 🤯' },
  { id: 'chill', label: 'Chill 😌' },
];

const GENRES = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
  'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
  'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western'];

// ── BOOT ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  setupNavScroll();
  setupSearch();
  renderAuthBar();
  renderCinemaToggle();
  route();
  window.addEventListener('hashchange', route);
});

// ── ROUTING ───────────────────────────────────
function route() {
  const page = location.hash.replace('#', '') || 'home';
  document.querySelectorAll('.nav-link').forEach(l =>
    l.classList.toggle('active', l.dataset.page === page));
  const app = document.getElementById('app');
  app.innerHTML = '';
  switch (page) {
    case 'home': renderHome(app); break;
    case 'browse': renderBrowse(app); break;
    case 'upcoming': renderUpcoming(app); break;
    case 'login': renderLogin(app); break;
    case 'register': renderRegister(app); break;
    case 'forgot': renderForgotPassword(app); break;
    case 'profile': renderProfile(app); break;
    default: renderHome(app);
  }
}

window.navigateTo = (p) => { location.hash = p; };

// ── CINEMA MODE TOGGLE ─────────────────────────
const INDUSTRIES = [
  { id: 'all_indian', label: '🇮🇳 All Indian' },
  { id: 'bollywood', label: '🎬 Bollywood' },
  { id: 'tollywood', label: '🎭 Tollywood' },
  { id: 'kollywood', label: '🎶 Kollywood' },
  { id: 'mollywood', label: '🌴 Mollywood' },
];

function renderCinemaToggle() {
  // Inject toggle into navbar right before nav-auth
  const navAuth = document.querySelector('.nav-auth');
  if (!navAuth || document.getElementById('cinema-toggle')) return;
  const wrap = document.createElement('div');
  wrap.id = 'cinema-toggle';
  wrap.innerHTML = `
    <div class="cinema-pill">
      <button class="cp-btn ${cinemaMode === 'indian' ? 'cp-active' : ''}" onclick="setCinemaMode('indian')">🇮🇳 Indian</button>
      <button class="cp-btn ${cinemaMode === 'hollywood' ? 'cp-active' : ''}" onclick="setCinemaMode('hollywood')">🎬 Hollywood</button>
    </div>`;
  navAuth.before(wrap);
}

function setCinemaMode(mode) {
  cinemaMode = mode;
  localStorage.setItem('cm_mode', mode);
  // Rebuild toggle UI
  const old = document.getElementById('cinema-toggle');
  if (old) old.remove();
  renderCinemaToggle();
  // Refresh home page if on home
  if (!location.hash || location.hash === '#home') navigateTo('home');
}

function switchIndustry(val) {
  industry = val;
  localStorage.setItem('cm_industry', val);
  if (!location.hash || location.hash === '#home') navigateTo('home');
}

window.setCinemaMode = setCinemaMode;
window.switchIndustry = switchIndustry;


function initParticles() {
  const c = document.getElementById('particles-canvas');
  const ctx = c.getContext('2d');
  let particles = [];
  function resize() { c.width = innerWidth; c.height = innerHeight; }
  resize(); window.addEventListener('resize', resize);
  class P {
    constructor() { this.reset(); }
    reset() {
      this.x = Math.random() * c.width;
      this.y = Math.random() * c.height;
      this.r = Math.random() * 1.5 + 0.3;
      this.vx = (Math.random() - 0.5) * 0.3;
      this.vy = (Math.random() - 0.5) * 0.3;
      this.alpha = Math.random() * 0.4 + 0.1;
    }
    update() {
      this.x += this.vx; this.y += this.vy;
      if (this.x < 0 || this.x > c.width || this.y < 0 || this.y > c.height) this.reset();
    }
    draw() {
      ctx.beginPath(); ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(229,9,20,${this.alpha})`; ctx.fill();
    }
  }
  for (let i = 0; i < 80; i++) particles.push(new P());
  function loop() { ctx.clearRect(0, 0, c.width, c.height); particles.forEach(p => { p.update(); p.draw(); }); requestAnimationFrame(loop); }
  loop();
}

// ── NAVBAR ────────────────────────────────────
function setupNavScroll() {
  window.addEventListener('scroll', () =>
    document.getElementById('navbar').classList.toggle('scrolled', scrollY > 50));
}

function renderAuthBar() {
  const el = document.getElementById('nav-auth');
  el.innerHTML = user
    ? `<button class="btn btn-ghost btn-sm" onclick="navigateTo('profile')">👤 ${user.username}</button>
       <button class="btn btn-primary btn-sm" onclick="logout()">Logout</button>`
    : `<button class="btn btn-outline btn-sm" onclick="navigateTo('login')">Login</button>
       <button class="btn btn-primary btn-sm" onclick="navigateTo('register')">Join Free</button>`;
}

function logout() {
  firebaseLogout().catch(() => { });
  user = null; localStorage.removeItem('cm_user');
  renderAuthBar(); toast('Logged out', 'ok'); navigateTo('home');
}

// ── SEARCH ────────────────────────────────────
let searchTimer;
function setupSearch() {
  const inp = document.getElementById('search-input');
  const dd = document.getElementById('search-dropdown');
  inp.addEventListener('input', () => {
    clearTimeout(searchTimer);
    const q = inp.value.trim();
    if (q.length < 2) { dd.classList.add('hidden'); return; }
    searchTimer = setTimeout(async () => {
      const mode = cinemaMode === 'indian' ? industry : 'hollywood';
      const res = await api(`/movies/search?q=${encodeURIComponent(q)}&industry=${mode}`);
      if (!res?.length) { dd.innerHTML = `<div class="si"><span style="color:var(--t3)">No results for "${q}"</span></div>`; dd.classList.remove('hidden'); return; }
      dd.innerHTML = res.map(m => `
        <div class="si" onclick="openModal(${m.id},'${esc(m.title)}');document.getElementById('search-dropdown').classList.add('hidden')">
          <img src="${m.poster_path || ph(40, 63)}" alt="">
          <div><div class="si-title">${m.title}</div>
          <div class="si-meta">${m.release_date?.slice(0, 4) || ''} · ★ ${m.vote_average?.toFixed(1) || 'N/A'}</div></div>
        </div>`).join('');
      dd.classList.remove('hidden');
    }, 350);
  });
  document.addEventListener('click', e => { if (!e.target.closest('.nav-search-wrap')) dd.classList.add('hidden'); });
}

// ── HOME ──────────────────────────────────────
async function renderHome(app) {
  app.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:80vh"><div class="spinner"></div></div>`;

  if (cinemaMode === 'indian') {
    await renderIndianHome(app);
    return;
  }

  // ── HOLLYWOOD MODE — fetch all in parallel ──────────
  const ind = 'hollywood';
  const [trending, nowPlaying, upcoming, ctx] = await Promise.all([
    api('/movies/trending'),
    api('/movies/now-playing'),
    api('/movies/upcoming'),
    api('/movies/contextual?industry=hollywood'),
  ]);

  const hero = trending?.[0];
  const bg = hero?.backdrop_path || hero?.poster_path || '';

  // ── PERSONALIZED SECTIONS (logged-in users) ──────────
  let hybridSection = '', forYouSection = '', justReleasedSection = '', classicsSection = '', byw = '';

  // Fetch classics + just-released always (no auth needed)
  const [classicsData, justRelData] = await Promise.all([
    api(`/recommendations/classics-like-trending?industry=${ind}`).catch(() => null),
    api(`/recommendations/just-released?username=${user?.username || ''}&industry=${ind}`).catch(() => null),
  ]);

  if (classicsData?.length) {
    classicsSection = `
      <div class="sec-head">
        <div class="sec-title">🎞️ Old Classics — Similar to What's Trending</div>
        <div class="sec-badge">Timeless Picks</div>
      </div>
      <div class="movie-row">${classicsData.map((m, i) => localCard(m, i)).join('')}</div>`;
  }
  if (justRelData?.length) {
    justReleasedSection = `
      <div class="sec-head">
        <div class="sec-title">🔥 Just Released For You</div>
        <div class="sec-badge new-badge">NEW</div>
      </div>
      <div class="movie-row">${justRelData.map((m, i) => tmdbCard(m, i)).join('')}</div>`;
  }

  if (user) {
    const activeMood = localStorage.getItem('cm_last_mood') || '';
    const [hybridData, forYouData] = await Promise.all([
      api(`/recommendations/hybrid?username=${user.username}&industry=${ind}&count=20&mood=${activeMood}&explain=true`).catch(() => null),
      api(`/recommendations/for-you?username=${user.username}&industry=${ind}&count=20`).catch(() => null),
    ]);
    if (hybridData?.length) {
      hybridSection = `
        <div class="hybrid-banner">
          <div class="hybrid-inner">
            <span class="hybrid-icon">🧠</span>
            <div>
              <div class="hybrid-title">AI Picks For ${user.username}</div>
              <div class="hybrid-sub">Trending · Taste · Similarity · Recency · Genre-balanced</div>
            </div>
          </div>
        </div>
        <div class="movie-row">${hybridData.map((m, i) => tmdbCard(m, i)).join('')}</div>`;
    }
    if (forYouData?.length) {
      const lastLiked = user.liked_movies?.filter(Boolean).slice(-1)[0];
      forYouSection = `
        <div class="byw-card">
          <div class="byw-header">
            <span class="byw-tag">🍿 BECAUSE YOU LOVED</span>
            <strong>${lastLiked || 'Your Favourites'}</strong>
          </div>
          <div class="movie-row">${forYouData.map((m, i) => localCard(m, i)).join('')}</div>
        </div>`;
    }
  }

  // ── Contextual / seasonal rows ──────────────────────
  const ctxBanner = ctx ? `
    <div class="ctx-banner">
      <div class="ctx-banner-inner">
        <span class="ctx-date">📅 Today — ${ctx.date}</span>
        <span class="ctx-label">${ctx.sections[0]?.label || ''}</span>
      </div>
    </div>` : '';

  const ctxRows = (ctx?.sections || []).map(sec => `
    <div class="sec-head"><div class="sec-title">${sec.label}</div></div>
    <div class="movie-row ctx-row" data-titles='${JSON.stringify(sec.movies.map(m => m.title))}'>
      ${sec.movies.map((m, i) => localCard(m, i)).join('')}
    </div>`).join('');

  // ── Restore last mood ────────────────────────────────
  const lastMood = localStorage.getItem('cm_last_mood');

  app.innerHTML = `
    ${buildHero(hero, bg)}
    <div class="mood-section">
      <div class="sec-title" style="padding:0">🎭 I'm feeling...</div>
      <div class="mood-grid">${MOODS.map(m => `<div class="mood-chip${lastMood === m.id ? ' active' : ''}" data-mood="${m.id}" onclick="pickMood('${m.id}')">${m.label}</div>`).join('')}</div>
      <div class="time-filter-row">
        <span class="tf-label">⏱️ Length:</span>
        <button class="tf-btn tf-active" id="tf-any" onclick="setTimeFilter('any')">📺 Any</button>
        <button class="tf-btn" id="tf-short" onclick="setTimeFilter('short')">⚡ Under 90 min</button>
        <button class="tf-btn" id="tf-normal" onclick="setTimeFilter('normal')">🎬 Under 2 hrs</button>
        <button class="tf-btn surprise-btn" onclick="doSurpriseMe()" title="Pick a random high-quality movie for you">🎲 Surprise Me!</button>
      </div>
    </div>
    <div id="mood-results"></div>
    ${hybridSection}
    ${justReleasedSection}
    ${forYouSection}
    ${ctxBanner}
    ${ctxRows}
    ${classicsSection}
    ${rowSection('🔥 Now Playing', nowPlaying)}
    ${rowSection('📈 Trending This Week', trending)}
    ${rowSection('🗓️ Coming Soon', upcoming)}
    <footer class="footer">Made with ❤️ for cinema lovers · <span>CINEMATE</span> © 2026</footer>`;

  // Batch-load contextual row posters
  if (ctx?.sections) {
    ctx.sections.forEach((sec, si) => {
      const rowEl = app.querySelectorAll('.ctx-row')[si];
      if (rowEl) batchLoadPosters(sec.movies.map(m => m.title), rowEl);
    });
  }
  // Batch-load classics posters
  if (classicsData?.length) {
    const classicsRowEl = app.querySelector('.movie-row:last-of-type');
    if (classicsRowEl) batchLoadPosters(classicsData.map(m => m.title), classicsRowEl);
  }
  // Restore last mood
  if (lastMood) pickMood(lastMood);
}

// ── INDIAN HOME ────────────────────────────────
async function renderIndianHome(app) {
  const reqIndustry = industry || 'all_indian';
  const [trending, nowPlaying, upcoming, topRated, ctx] = await Promise.all([
    api(`/movies/indian-trending?industry=${reqIndustry}`),
    api(`/movies/indian-now-playing?industry=${reqIndustry}`),
    api(`/movies/indian-upcoming?industry=${reqIndustry}`),
    api(`/movies/indian-top-rated?industry=${reqIndustry}`),
    api(`/movies/contextual?industry=${reqIndustry}`),
  ]);

  const hero = trending?.[0];
  const bg = hero?.backdrop_path || hero?.poster_path || '';

  // ── PERSONALIZED SECTIONS ──────────────────────────────
  let hybridSection = '', forYouSection = '', justReleasedSection = '';

  const justRelData = await api(`/recommendations/just-released?username=${user?.username || ''}&industry=${reqIndustry}`).catch(() => null);
  if (justRelData?.length) {
    justReleasedSection = `
      <div class="sec-head">
        <div class="sec-title">🔥 Just Released For You</div>
        <div class="sec-badge new-badge">NEW</div>
      </div>
      <div class="movie-row">${justRelData.map((m, i) => tmdbCard(m, i)).join('')}</div>`;
  }

  if (user) {
    const [hybridData, forYouData] = await Promise.all([
      api(`/recommendations/hybrid?username=${user.username}&industry=${reqIndustry}&count=20`).catch(() => null),
      api(`/recommendations/for-you?username=${user.username}&industry=${reqIndustry}&count=20`).catch(() => null),
    ]);
    if (hybridData?.length) {
      hybridSection = `
        <div class="hybrid-banner">
          <div class="hybrid-inner">
            <span class="hybrid-icon">🧠</span>
            <div>
              <div class="hybrid-title">AI Picks For ${user.username}</div>
              <div class="hybrid-sub">Trending · Taste · Similarity · Recency combined</div>
            </div>
          </div>
        </div>
        <div class="movie-row">${hybridData.map((m, i) => tmdbCard(m, i)).join('')}</div>`;
    }
    if (forYouData?.length) {
      const lastLiked = user.liked_movies?.filter(Boolean).slice(-1)[0];
      forYouSection = `
        <div class="byw-card">
          <div class="byw-header">
            <span class="byw-tag">🍿 BECAUSE YOU LOVED</span>
            <strong>${lastLiked || 'Your Favourites'}</strong>
          </div>
          <div class="movie-row">${forYouData.map((m, i) => localCard(m, i)).join('')}</div>
        </div>`;
    }
  }

  const ctxBanner = ctx ? `
    <div class="ctx-banner" style="margin-top:2rem">
      <div class="ctx-banner-inner">
        <span class="ctx-date">📅 Today — ${ctx.date}</span>
        <span class="ctx-label">${ctx.sections[0]?.label || ''}</span>
      </div>
    </div>` : '';

  const ctxRows = (ctx?.sections || []).map((sec, si) => `
    <div class="sec-head"><div class="sec-title">${sec.label}</div></div>
    <div class="movie-row ctx-row" data-titles='${JSON.stringify(sec.movies.map(m => m.title))}'>
      ${sec.movies.map((m, i) => localCard(m, i)).join('')}
    </div>`).join('');

  const lastMood = localStorage.getItem('cm_last_mood');

  app.innerHTML = `
    ${buildHero(hero, bg, '🇮🇳')}
    <div class="indian-banner">
      <div class="indian-banner-inner">
        <span class="indian-flag">🇮🇳</span>
        <div>
          <div class="indian-title">Indian Cinema</div>
          <div class="indian-sub">Bollywood · Tollywood · Kollywood · Mollywood · Sandalwood</div>
        </div>
      </div>
    </div>
    <div class="mood-section">
      <div class="sec-title" style="padding:0">🎭 I'm feeling...</div>
      <div class="mood-grid">${MOODS.map(m => `<div class="mood-chip${lastMood === m.id ? ' active' : ''}" data-mood="${m.id}" onclick="pickMood('${m.id}')">${m.label}</div>`).join('')}</div>
      <div class="time-filter-row">
        <span class="tf-label">⏱️ Length:</span>
        <button class="tf-btn tf-active" id="tf-any" onclick="setTimeFilter('any')">📺 Any</button>
        <button class="tf-btn" id="tf-short" onclick="setTimeFilter('short')">⚡ Under 90 min</button>
        <button class="tf-btn" id="tf-normal" onclick="setTimeFilter('normal')">🎬 Under 2 hrs</button>
      </div>
    </div>
    <div id="mood-results"></div>
    ${hybridSection}
    ${justReleasedSection}
    ${forYouSection}
    ${ctxBanner}
    ${ctxRows}
    ${rowSection('🔥 Trending in India', trending)}
    ${rowSection('🎬 Now Showing in Theatres', nowPlaying)}
    ${rowSection('⭐ All-Time Indian Greats', topRated)}
    ${rowSection('🗓️ Coming Soon', upcoming)}
    <footer class="footer">Celebrating Indian Cinema · <span>CINEMATE</span> © 2026</footer>`;

  if (ctx?.sections) {
    ctx.sections.forEach((sec, si) => {
      const rowEl = app.querySelectorAll('.ctx-row')[si];
      if (rowEl) batchLoadPosters(sec.movies.map(m => m.title), rowEl);
    });
  }
  if (lastMood) pickMood(lastMood);
}

function buildHero(m, bg) {
  if (!m) return '';
  return `
    <div class="hero">
      <div class="hero-bg" style="background-image:url('${bg}')"></div>
      <div class="hero-overlay">
        <div class="hero-badge">🔥 Trending Now</div>
        <h1 class="hero-title">${m.title}</h1>
        <div class="hero-meta">
          <span class="hero-rating">★ ${m.vote_average?.toFixed(1) || 'N/A'}</span>
          <span style="color:var(--t2)">${m.release_date?.slice(0, 4) || ''}</span>
        </div>
        <p class="hero-desc">${(m.overview || '').substring(0, 160)}...</p>
        <div class="hero-actions">
          <button class="btn btn-primary" onclick="openModal(${m.id},'${esc(m.title)}')">▶ Watch Trailer</button>
          <button class="btn btn-ghost" onclick="addToWatchlist('${esc(m.title)}')">+ Watchlist</button>
        </div>
      </div>
    </div>`;
}

function rowSection(title, movies) {
  if (!movies?.length) return '';
  return `<div class="sec-head"><div class="sec-title">${title}</div></div>
    <div class="movie-row">${movies.map((m, i) => tmdbCard(m, i)).join('')}</div>`;
}

// ── MOOD ──────────────────────────────────────
let currentTimeFilter = 'any';

window.setTimeFilter = function (filter) {
  currentTimeFilter = filter;
  localStorage.setItem('cm_time_filter', filter);
  document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('tf-active'));
  const btn = document.getElementById(`tf-${filter}`);
  if (btn) btn.classList.add('tf-active');
  // Re-run mood if one is selected
  if (currentMood) pickMood(currentMood);
};

// ── SURPRISE ME ────────────────────────────────
window.doSurpriseMe = async function () {
  const btn = document.querySelector('.surprise-btn');
  if (btn) { btn.textContent = '⏳ Picking...'; btn.disabled = true; }
  try {
    const ind = cinemaMode === 'indian' ? industry : 'all';
    const uname = user?.username || '';
    const movie = await api(`/recommendations/surprise?username=${uname}&industry=${ind}`);
    if (movie?.title) {
      toast(`🎲 Surprise: ${movie.title}`, 'ok');
      openModal(movie.id || 0, movie.title);
    } else {
      toast('Could not find a surprise pick — try again!', 'info');
    }
  } catch (e) {
    toast('Surprise pick failed', 'err');
  } finally {
    if (btn) { btn.textContent = '🎲 Surprise Me!'; btn.disabled = false; }
  }
};

window.pickMood = async function (mood) {
  document.querySelectorAll('.mood-chip').forEach(c => c.classList.toggle('active', c.dataset.mood === mood));
  currentMood = mood;
  localStorage.setItem('cm_last_mood', mood);
  const el = document.getElementById('mood-results');
  el.innerHTML = `<div class="sec-head"><div class="sec-title">🎬 Movies for your mood</div></div><div class="movie-row">${Array(8).fill(`<div class="skel" style="flex:0 0 175px;aspect-ratio:2/3"></div>`).join('')}</div>`;
  const reqIndustry = cinemaMode === 'indian' ? industry : 'hollywood';
  const res = await api('/recommendations/mood', 'POST', { mood, count: 20, industry: reqIndustry });
  if (res?.length) {
    const moodLabel = MOODS.find(m => m.id === mood)?.label;
    el.innerHTML = `<div class="sec-head"><div class="sec-title">🎬 ${moodLabel} Picks</div></div>
      <div class="movie-row">${res.map((m, i) => localCard(m, i)).join('')}</div>`;
    batchLoadPosters(res.map(m => m.title), el);
  }
};

// ── BROWSE ────────────────────────────────────
async function renderBrowse(app) {
  const titles = await api('/movies/all-titles') || [];
  app.innerHTML = `
    <div style="padding-top:80px">
      <div class="filter-bar">
        <select id="b-type" onchange="updateBrowseInputs()">
          <option value="genre">Genre Based</option>
          <option value="movie">Movie Based</option>
          <option value="mood">Mood Based</option>
        </select>
        <select id="b-genre">
          ${GENRES.map(g => `<option value="${g}">${g}</option>`).join('')}
        </select>
        <select id="b-movie" style="display:none;max-width:280px">
          ${titles.map(t => `<option value="${esc(t)}">${t}</option>`).join('')}
        </select>
        <select id="b-mood" style="display:none">
          ${MOODS.map(m => `<option value="${m.id}">${m.label}</option>`).join('')}
        </select>
        <input type="number" id="b-rating" placeholder="Min Rating (1-10)" min="1" max="10" value="7" style="width:170px">
        <button class="btn btn-primary btn-sm" onclick="runBrowse()">Get Recommendations</button>
      </div>
      <div id="browse-results" class="movie-grid">${skelGrid(12)}</div>
      <footer class="footer">CINEMATE © 2026</footer>
    </div>`;
  runBrowse();
}

window.updateBrowseInputs = function () {
  const t = document.getElementById('b-type').value;
  document.getElementById('b-genre').style.display = t === 'genre' ? '' : 'none';
  document.getElementById('b-movie').style.display = t === 'movie' ? '' : 'none';
  document.getElementById('b-mood').style.display = t === 'mood' ? '' : 'none';
};

window.runBrowse = async function () {
  const g = document.getElementById('browse-results');
  if (!g) return;
  g.innerHTML = skelGrid(12);
  const type = document.getElementById('b-type').value;
  const rating = parseFloat(document.getElementById('b-rating').value) || 7;
  const reqIndustry = cinemaMode === 'indian' ? industry : 'hollywood';
  let res;
  if (type === 'genre') {
    const genre = document.getElementById('b-genre').value;
    res = await api('/recommendations/genre', 'POST', { genres: [genre], min_rating: rating, count: 40, industry: reqIndustry });
  } else if (type === 'movie') {
    const title = document.getElementById('b-movie').value;
    res = await api('/recommendations/movie', 'POST', { movie_title: title, count: 40, industry: reqIndustry });
  } else {
    const mood = document.getElementById('b-mood').value;
    res = await api('/recommendations/mood', 'POST', { mood, count: 40, industry: reqIndustry });
  }
  g.innerHTML = res?.length ? res.map((m, i) => localCard(m, i, true)).join('') : '<p style="color:var(--t3);padding:2rem">No results found.</p>';
  if (res?.length) batchLoadPosters(res.map(m => m.title), g);
};

// ── UPCOMING MOVIES PAGE ──────────────────────────────
async function renderUpcoming(app) {
  // Show skeleton loader
  app.innerHTML = `<div class="upcoming-page" style="padding-top:90px">
    <div class="upcoming-hero">
      <div class="upcoming-hero-inner">
        <div class="upcoming-badge">🗓️ COMING SOON</div>
        <h1 class="upcoming-hero-title">Upcoming <span class="upcoming-accent">Blockbusters</span></h1>
        <p class="upcoming-hero-sub">The most anticipated movies dropping soon — be the first to know.</p>
        <div class="upcoming-toggle-wrap">
          <div class="upcoming-mode-pill" id="upcoming-mode-pill">
            <button class="umode-btn ${cinemaMode === 'indian' ? 'umode-active' : ''}" id="umode-indian" onclick="setUpcomingMode('indian')">🇮🇳 Indian</button>
            <button class="umode-btn ${cinemaMode === 'hollywood' ? 'umode-active' : ''}" id="umode-hollywood" onclick="setUpcomingMode('hollywood')">🎬 Hollywood</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Industry sub-filter (Indian only) -->
    <div class="upcoming-industry-bar" id="upcoming-industry-bar" style="${cinemaMode === 'indian' ? '' : 'display:none'}">
      ${INDUSTRIES.map(ind => `<button class="uid-btn ${industry === ind.id ? 'uid-active' : ''}" onclick="setUpcomingIndustry('${ind.id}')">${ind.label}</button>`).join('')}
    </div>

    <!-- Sort / filter bar -->
    <div class="upcoming-filter-bar">
      <div class="ufb-left">
        <span class="ufb-label">Sort by:</span>
        <button class="ufb-btn ufb-active" id="usort-date" onclick="setUpcomingSort('date')">📅 Release Date</button>
        <button class="ufb-btn" id="usort-pop" onclick="setUpcomingSort('popularity')">🔥 Popularity</button>
      </div>
      <div class="ufb-right">
        <span id="upcoming-count-badge" class="ufb-count">Loading...</span>
      </div>
    </div>

    <!-- Grid of upcoming movies -->
    <div id="upcoming-grid" class="upcoming-grid">${skelUpcoming(12)}</div>
    <footer class="footer">🎬 Stay tuned for what's dropping next · <span>CINEMATE</span> © 2026</footer>
  </div>`;

  // expose sort state
  window._upcomingSort = 'date';
  window._upcomingData = [];

  await loadUpcomingMovies();
}

function skelUpcoming(n) {
  return Array(n).fill(`
    <div class="uc-skel">
      <div class="skel" style="aspect-ratio:2/3;border-radius:16px;height:260px"></div>
      <div class="skel" style="height:14px;border-radius:8px;margin-top:10px;width:80%"></div>
      <div class="skel" style="height:10px;border-radius:8px;margin-top:6px;width:50%"></div>
    </div>`).join('');
}

async function loadUpcomingMovies() {
  const grid = document.getElementById('upcoming-grid');
  if (!grid) return;
  grid.innerHTML = skelUpcoming(12);

  let movies = [];
  if (cinemaMode === 'indian') {
    movies = await api(`/movies/indian-upcoming?industry=${industry}`).catch(() => []);
  } else {
    movies = await api('/movies/upcoming').catch(() => []);
  }

  // ── Client-side safety: only keep genuinely future-dated movies ──
  const todayStr = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
  movies = (movies || []).filter(m => {
    if (!m.release_date) return true;   // keep date-TBA movies
    return m.release_date >= todayStr;
  });

  window._upcomingData = movies;
  renderUpcomingGrid();
}

function renderUpcomingGrid() {
  const grid = document.getElementById('upcoming-grid');
  if (!grid) return;
  let movies = [...(window._upcomingData || [])];

  // Sort
  if (window._upcomingSort === 'date') {
    movies.sort((a, b) => (a.release_date || '9999') < (b.release_date || '9999') ? -1 : 1);
  } else {
    movies.sort((a, b) => (b.vote_average || 0) - (a.vote_average || 0));
  }

  const badge = document.getElementById('upcoming-count-badge');
  if (badge) badge.textContent = `${movies.length} movies`;

  if (!movies.length) {
    grid.innerHTML = `<div class="upcoming-empty">
      <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
      <h3>No upcoming movies found</h3>
      <p style="color:var(--t2);margin-top:.5rem">Try switching cinema mode or industry.</p>
    </div>`;
    return;
  }

  grid.innerHTML = movies.map((m, i) => upcomingCard(m, i)).join('');
}

function upcomingCard(m, i) {
  const poster = m.poster_path || ph(220, 330);
  const rating = m.vote_average ? m.vote_average.toFixed(1) : 'TBD';
  const releaseDate = m.release_date || '';
  const fullDate = releaseDate ? formatReleaseDate(releaseDate) : 'Date TBA';
  const countdown = releaseDate ? getDaysUntil(releaseDate) : null;

  let countdownHtml;
  if (countdown === null) {
    countdownHtml = `<span class="uc-tba">Date TBA</span>`;
  } else if (countdown === 0) {
    countdownHtml = `<span class="uc-today">🎉 Releasing Today!</span>`;
  } else if (countdown <= 7) {
    countdownHtml = `<span class="uc-soon">🔥 ${countdown}d to go!</span>`;
  } else if (countdown <= 30) {
    countdownHtml = `<span class="uc-countdown">⏳ ${countdown} days away</span>`;
  } else {
    // For far-future movies, show month+year label
    const months = Math.round(countdown / 30);
    countdownHtml = `<span class="uc-far">📅 ${months} month${months > 1 ? 's' : ''} away</span>`;
  }

  const d = Math.min(i * 0.035, 0.5);
  return `
    <div class="uc" style="animation-delay:${d}s" onclick="openModal(${m.id || 0},'${esc(m.title)}')">
      <div class="uc-poster-wrap">
        <img class="uc-poster" src="${poster}" loading="lazy" alt="${m.title}">
        <div class="uc-overlay">
          <div class="uc-play">▶</div>
          <div class="uc-actions">
            <button class="mc-btn" onclick="event.stopPropagation();toggleLike('${esc(m.title)}')" title="Like">♥</button>
            <button class="mc-btn" onclick="event.stopPropagation();addToWatchlist('${esc(m.title)}')" title="Watchlist">🎯</button>
          </div>
        </div>
        <div class="uc-badge-wrap">
          ${countdownHtml}
        </div>
      </div>
      <div class="uc-info">
        <div class="uc-title">${m.title}</div>
        <div class="uc-meta">
          <span class="uc-date">📅 ${fullDate}</span>
          ${rating !== 'TBD' ? `<span class="uc-rating">★ ${rating}</span>` : `<span class="uc-rating-tbd">★ TBD</span>`}
        </div>
        ${m.overview ? `<p class="uc-overview">${m.overview.slice(0, 110)}...</p>` : ''}
      </div>
    </div>`;
}

function formatReleaseDate(dateStr) {
  if (!dateStr) return 'TBA';
  try {
    return new Date(dateStr).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
  } catch { return dateStr; }
}

function getDaysUntil(dateStr) {
  if (!dateStr) return null;
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const rel = new Date(dateStr); rel.setHours(0, 0, 0, 0);
  return Math.round((rel - today) / (1000 * 60 * 60 * 24));
}

window.setUpcomingMode = function (mode) {
  cinemaMode = mode;
  localStorage.setItem('cm_mode', mode);
  // Rebuild global toggle too
  const old = document.getElementById('cinema-toggle');
  if (old) old.remove();
  renderCinemaToggle();
  // Update pill buttons
  document.querySelectorAll('.umode-btn').forEach(b => {
    b.classList.toggle('umode-active', b.id === `umode-${mode}`);
  });
  const bar = document.getElementById('upcoming-industry-bar');
  if (bar) bar.style.display = mode === 'indian' ? '' : 'none';
  loadUpcomingMovies();
};

window.setUpcomingIndustry = function (ind) {
  industry = ind;
  localStorage.setItem('cm_industry', ind);
  document.querySelectorAll('.uid-btn').forEach(b => {
    b.classList.toggle('uid-active', b.textContent.trim().includes(INDUSTRIES.find(x => x.id === ind)?.label.split(' ')[1] || ''));
  });
  // re-mark active more reliably
  document.querySelectorAll('.uid-btn').forEach((b, idx) => {
    b.classList.toggle('uid-active', INDUSTRIES[idx]?.id === ind);
  });
  loadUpcomingMovies();
};

window.setUpcomingSort = function (sort) {
  window._upcomingSort = sort;
  document.getElementById('usort-date')?.classList.toggle('ufb-active', sort === 'date');
  document.getElementById('usort-pop')?.classList.toggle('ufb-active', sort === 'popularity');
  renderUpcomingGrid();
};

// (Analytics Removed)

// ── MOVIE MODAL ───────────────────────────────
window.openModal = async function (id, title) {
  const modal = document.getElementById('movie-modal');
  const body = document.getElementById('modal-body');
  modal.classList.remove('hidden');
  modalMovieId = id;
  body.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:300px"><div class="spinner"></div></div>`;

  // Track watch history
  if (user && title) {
    api('/user/history', 'POST', { username: user.username, movie_title: title }).catch(() => { });
  }

  // Country for OTT providers: IN for Indian cinema, US otherwise
  const ottCountry = cinemaMode === 'indian' ? 'IN' : 'US';

  // Fetch everything in parallel
  let movieMeta = null;
  const [searchRes, details] = await Promise.all([
    api(`/movies/search?q=${encodeURIComponent(title)}`),
    id ? api(`/movies/${id}/details`) : Promise.resolve(null)
  ]);
  movieMeta = searchRes?.[0] || null;
  if (!id && movieMeta) { id = movieMeta.id; }
  const finalDetails = details || (id ? await api(`/movies/${id}/details`) : null);

  // Fetch OTT providers now that we have the id
  const ottData = id
    ? await api(`/movies/${id}/watch-providers?country=${ottCountry}&title=${encodeURIComponent(title)}`).catch(() => null)
    : null;
  const providers = ottData?.providers || [];
  const tmdbLink = ottData?.tmdb_link || '';

  const backdrop = movieMeta?.backdrop_path || movieMeta?.poster_path || '';
  const trailer = finalDetails?.trailers?.[0];
  const isLiked = user?.liked_movies?.includes(title);
  const inWL = user?.watchlist?.includes(title);
  const myRating = user?.ratings?.[title] || 0;

  // ── OTT badges HTML ──────────────────────────────────────
  let ottHtml = '';
  if (providers.length) {
    ottHtml = `
      <div class="ott-section">
        <div class="ott-label">🎬 Watch On</div>
        <div class="ott-row">
          ${providers.map(p => `
            <a class="ott-btn" href="${p.url}" target="_blank" rel="noopener"
               style="background:${p.color}22;border-color:${p.color}55;"
               title="Watch on ${p.name}">
              ${p.logo ? `<img src="${p.logo}" alt="${p.name}" class="ott-logo">` : ''}
              <span class="ott-name">${p.name}</span>
              <span class="ott-watch-text">Watch ↗</span>
            </a>`).join('')}
          ${tmdbLink ? `<a class="ott-btn ott-more" href="${tmdbLink}" target="_blank" rel="noopener" title="All providers on JustWatch">
            <span class="ott-name">More Options</span>
            <span class="ott-watch-text">↗</span>
          </a>` : ''}
        </div>
      </div>`;
  } else if (id) {
    // Not on any subscription platform — show JustWatch fallback
    const jwUrl = `https://www.justwatch.com/in/search?q=${encodeURIComponent(title)}`;
    ottHtml = `
      <div class="ott-section">
        <div class="ott-label">🎬 Where to Watch</div>
        <div class="ott-row">
          <a class="ott-btn ott-justwatch" href="${jwUrl}" target="_blank" rel="noopener">
            <span class="ott-name">🔍 Check JustWatch</span>
            <span class="ott-watch-text">↗</span>
          </a>
        </div>
      </div>`;
  }

  body.innerHTML = `
    <div class="modal-hero"><img src="${backdrop || ph(960, 280)}" alt="">${backdrop ? '<div class="modal-hero-veil"></div>' : ''}</div>
    <div class="modal-body">
      <div style="display:flex;gap:12px;margin-bottom:.5rem;flex-wrap:wrap">
        <button class="btn btn-primary btn-sm ${isLiked ? '' : 'btn-outline'}" onclick="toggleLike('${esc(title)}')">
          ${isLiked ? '❤️ Liked' : '🤍 Like'}
        </button>
        <button class="btn btn-sm ${inWL ? 'btn-primary' : 'btn-outline'}" onclick="addToWatchlist('${esc(title)}')">
          ${inWL ? '✅ In Watchlist' : '+ Watchlist'}
        </button>
      </div>
      <h2 class="modal-title">${movieMeta?.title || title}</h2>
      <div class="modal-meta-row">
        <span class="mc-rating">★ ${movieMeta?.vote_average?.toFixed(1) || 'N/A'}</span>
        <span>${movieMeta?.release_date || ''}</span>
      </div>
      <div class="star-row" id="star-row-${id}">
        <span style="color:var(--t2);font-size:.85rem;margin-right:4px">Your rating:</span>
        ${[1, 2, 3, 4, 5].map(n => `<span class="star ${myRating >= n ? 'lit' : ''}" onclick="rateMovie('${esc(title)}',${n},${id})" data-val="${n}">★</span>`).join('')}
      </div>
      <p class="modal-overview">${movieMeta?.overview || ''}</p>

      ${ottHtml}

      ${trailer ? `<h3 class="modal-sec">🎬 Official Trailer</h3>
        <div class="trailer-wrapper"><iframe src="https://www.youtube.com/embed/${trailer.key}?autoplay=0" allowfullscreen></iframe></div>` : ''}

      ${finalDetails?.cast?.length ? `<h3 class="modal-sec">🎭 Cast</h3>
        <div class="cast-scroll">
          ${finalDetails.cast.map(c => `
            <div class="cast-card">
              <img src="${c.profile_path ? 'https://image.tmdb.org/t/p/w185' + c.profile_path : ph(66, 66)}" loading="lazy" alt="${c.name}">
              <div class="cast-card-name">${c.name}</div>
              <div class="cast-card-char">${c.character || ''}</div>
            </div>`).join('')}
        </div>`: ''}

      ${finalDetails?.similar?.length ? `<h3 class="modal-sec" style="margin-top:1.5rem">🍿 More Like This</h3>
        <div class="similar-scroll">
          ${finalDetails.similar.map(m => `
            <div class="sim-card" onclick="openModal(${m.id},'${esc(m.title)}')">
              <img src="${m.poster_path || ph(130, 195)}" loading="lazy" alt="${m.title}">
              <div class="sim-card-name">${m.title}</div>
            </div>`).join('')}
        </div>`: ''}
    </div>`;
};

window.closeModal = () => document.getElementById('movie-modal').classList.add('hidden');

// Expose auth functions to global scope (required because main.js is an ES module)
window.doLogin = (...a) => doLogin(...a);
window.doRegister = (...a) => doRegister(...a);
window.doForgotPassword = (...a) => doForgotPassword(...a);
window.doGoogleSignIn = (...a) => doGoogleSignIn(...a);
window.logout = (...a) => logout(...a);

// ── AUTH PAGES (Firebase) ─────────────────────
function authCard(title, body, footer = '') {
  return `<div class="auth-wrap"><div class="auth-card">
    <h2>${title}</h2>${body}${footer}
  </div></div>`;
}

function renderLogin(app) {
  app.innerHTML = authCard('Sign In', `
    <div class="fg"><label>Email</label><input id="l-e" type="email" placeholder="you@example.com"></div>
    <div class="fg"><label>Password</label>
      <input id="l-p" type="password" placeholder="Your password">
      <div style="text-align:right;margin-top:6px">
        <a onclick="navigateTo('forgot')" style="font-size:.8rem;color:var(--red);cursor:pointer">Forgot password?</a>
      </div>
    </div>
    <button class="btn btn-primary" style="width:100%;padding:14px;font-size:1rem;margin-bottom:.8rem" onclick="doLogin()">Sign In</button>
    <div class="auth-divider">or</div>
    <button class="btn-google" onclick="doGoogleSignIn()">🔵 Continue with Google</button>`,
    `<div class="auth-switch">New here? <a onclick="navigateTo('register')" style="cursor:pointer">Create account</a></div>`);
}

async function doLogin() {
  const email = document.getElementById('l-e').value.trim();
  const pw = document.getElementById('l-p').value;
  if (!email || !pw) { toast('Fill all fields', 'err'); return; }
  try {
    const fbUser = await firebaseLogin(email, pw);
    await syncFirebaseUser(fbUser);
  } catch (e) { toast(friendlyFbError(e), 'err'); }
}

async function doGoogleSignIn() {
  try {
    const fbUser = await firebaseLoginGoogle();
    await syncFirebaseUser(fbUser);
  } catch (e) { toast(friendlyFbError(e), 'err'); }
}

function renderRegister(app) {
  app.innerHTML = authCard('Create Account', `
    <div class="fg"><label>Username</label><input id="r-u" placeholder="Choose a username"></div>
    <div class="fg"><label>Email</label><input id="r-e" type="email" placeholder="you@example.com"></div>
    <div class="fg"><label>Password <span style="color:var(--t3);font-weight:400">(min 6 chars)</span></label>
      <input id="r-p" type="password" placeholder="Strong password"></div>
    <div class="fg"><label>Favorite Genres <span style="color:var(--t3);font-weight:400">(Ctrl/Cmd for multiple)</span></label>
      <select id="r-g" multiple style="height:110px">
        ${GENRES.map(g => `<option value="${g}">${g}</option>`).join('')}
      </select></div>
    <button class="btn btn-primary" style="width:100%;padding:14px;font-size:1rem;margin-bottom:.8rem" onclick="doRegister()">Create Account</button>
    <div class="auth-divider">or</div>
    <button class="btn-google" onclick="doGoogleSignIn()">🔵 Continue with Google</button>`,
    `<div class="auth-switch">Already a member? <a onclick="navigateTo('login')" style="cursor:pointer">Sign In</a></div>`);
}

async function doRegister() {
  const username = document.getElementById('r-u').value.trim();
  const email = document.getElementById('r-e').value.trim();
  const pw = document.getElementById('r-p').value;
  const genres = [...document.getElementById('r-g').selectedOptions].map(o => o.value);
  if (!username || !email || !pw) { toast('Fill all fields', 'err'); return; }
  if (pw.length < 6) { toast('Password must be at least 6 characters', 'err'); return; }
  try {
    const fbUser = await firebaseRegister(email, pw, username);
    await syncFirebaseUser(fbUser, username, genres);
  } catch (e) { toast(friendlyFbError(e), 'err'); }
}

function renderForgotPassword(app) {
  app.innerHTML = authCard('Reset Password', `
    <p style="color:var(--t2);font-size:.9rem;margin-bottom:1.5rem">We'll send a reset link to your email address.</p>
    <div class="fg"><label>Email</label><input id="fp-e" type="email" placeholder="you@example.com"></div>
    <button class="btn btn-primary" style="width:100%;padding:14px;font-size:1rem" onclick="doForgotPassword()">Send Reset Link ✉️</button>`,
    `<div class="auth-switch"><a onclick="navigateTo('login')" style="cursor:pointer">← Back to Sign In</a></div>`);
}

async function doForgotPassword() {
  const email = document.getElementById('fp-e').value.trim();
  if (!email) { toast('Enter your email address', 'err'); return; }
  try {
    await firebaseForgotPassword(email);
    toast('✅ Reset email sent! Check your inbox.', 'ok');
    setTimeout(() => navigateTo('login'), 2500);
  } catch (e) { toast(friendlyFbError(e), 'err'); }
}

/** After Firebase auth, sync with our backend + update state. */
async function syncFirebaseUser(fbUser, preferredUsername = '', genres = []) {
  const idToken = await fbUser.getIdToken();
  const username = preferredUsername || fbUser.displayName || fbUser.email?.split('@')[0] || '';
  try {
    const res = await api('/auth/firebase', 'POST', { id_token: idToken, username, genres });
    user = res; localStorage.setItem('cm_user', JSON.stringify(res));
    renderAuthBar(); toast(`Welcome, ${res.username}! 🎬`, 'ok'); navigateTo('home');
  } catch (e) { toast(e.message || 'Sync with server failed', 'err'); }
}

function friendlyFbError(e) {
  const c = e.code || '';
  if (c.includes('email-already')) return 'That email is already registered.';
  if (c.includes('wrong-password')) return 'Incorrect password.';
  if (c.includes('user-not-found')) return 'No account found with that email.';
  if (c.includes('invalid-email')) return 'Please enter a valid email address.';
  if (c.includes('weak-password')) return 'Password is too weak (min 6 chars).';
  if (c.includes('popup-closed')) return 'Google sign-in was cancelled.';
  if (c.includes('network')) return 'Network error. Check your connection.';
  return e.message || 'Authentication failed.';
}

// ── PROFILE ───────────────────────────────────
async function renderProfile(app) {
  if (!user) { navigateTo('login'); return; }
  const data = await api(`/user/${user.username}/data`);
  Object.assign(user, data); localStorage.setItem('cm_user', JSON.stringify(user));

  app.innerHTML = `
    <div class="profile-wrap">
      <div class="profile-head">
        <div class="profile-av">${user.username[0].toUpperCase()}</div>
        <div>
          <h2 style="font-size:1.8rem;font-weight:800">${user.username}</h2>
          <p style="color:var(--t2)">${user.genres?.join(' · ') || 'No genres set'}</p>
        </div>
      </div>
      <div class="tab-bar">
        <div class="tab active" onclick="switchTab('likes')">❤️ Liked (${data.liked_movies?.filter(Boolean).length || 0})</div>
        <div class="tab" onclick="switchTab('watchlist')">🎯 Watchlist (${data.watchlist?.filter(Boolean).length || 0})</div>
        <div class="tab" onclick="switchTab('rated')">⭐ Rated (${Object.keys(data.ratings || {}).length})</div>
      </div>
      <div id="tab-content"></div>
    </div>
    <footer class="footer">CINEMATE © 2026</footer>`;

  switchTab('likes');
}

window.switchTab = async function (tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.textContent.toLowerCase().includes(tab.split('s')[0])));
  const el = document.getElementById('tab-content');
  const data = await api(`/user/${user.username}/data`);
  const renderList = async (titles) => {
    const valid = (titles || []).filter(Boolean);
    if (!valid.length) { el.innerHTML = `<p style="color:var(--t3);padding:2rem">Nothing here yet.</p>`; return; }
    // Show skeletons immediately
    el.innerHTML = `<div class="movie-grid">${skelGrid(valid.length)}</div>`;
    // Batch fetch all at once
    const batchMap = await api('/movies/batch-search', 'POST', { titles: valid }).catch(() => ({}));
    const cards = valid.map((t, i) => {
      const m = batchMap[t] || { title: t, poster_path: null, vote_average: 0, id: 0 };
      return tmdbCard(m, i);
    });
    el.innerHTML = `<div class="movie-grid">${cards.join('')}</div>`;
  };

  if (tab === 'likes') renderList(data.liked_movies);
  if (tab === 'watchlist') renderList(data.watchlist);
  if (tab === 'rated') {
    const entries = Object.entries(data.ratings || {});
    if (!entries.length) { el.innerHTML = `<p style="color:var(--t3);padding:2rem">You haven't rated anything yet.</p>`; return; }
    el.innerHTML = `<div style="display:flex;flex-direction:column;gap:12px;max-width:600px">
      ${entries.map(([t, r]) => `<div style="display:flex;align-items:center;justify-content:space-between;background:var(--bg3);border:1px solid var(--border);border-radius:var(--r-sm);padding:14px 20px">
        <span style="font-weight:700">${t}</span>
        <span style="color:var(--gold)">${'★'.repeat(r)}${'☆'.repeat(5 - r)}</span>
      </div>`).join('')}
    </div>`;
  }
};

// ── LIKE / WATCHLIST / RATE ────────────────────
window.toggleLike = async function (title) {
  if (!user) { toast('Sign in to like movies', 'info'); navigateTo('login'); return; }
  try {
    const res = await api('/user/likes', 'POST', { username: user.username, movie_title: title });
    user.liked_movies = res.liked_movies; localStorage.setItem('cm_user', JSON.stringify(user));
    toast(res.is_liked ? `❤️ Liked: ${title}` : `Removed from likes`, 'ok');
    // Re-open modal to refresh buttons
    if (modalMovieId) { const cur = document.getElementById('modal-body'); if (cur) openModal(modalMovieId, title); }
  } catch (e) { toast('Failed', 'err'); }
};

window.addToWatchlist = async function (title) {
  if (!user) { toast('Sign in to add to watchlist', 'info'); navigateTo('login'); return; }
  try {
    const res = await api('/user/watchlist', 'POST', { username: user.username, movie_title: title });
    user.watchlist = res.watchlist; localStorage.setItem('cm_user', JSON.stringify(user));
    toast(res.in_watchlist ? `🎯 Added to Watchlist` : `Removed from Watchlist`, 'ok');
  } catch (e) { toast('Failed', 'err'); }
};

window.rateMovie = async function (title, rating, id) {
  if (!user) { toast('Sign in to rate movies', 'info'); navigateTo('login'); return; }
  try {
    const res = await api('/user/rate', 'POST', { username: user.username, movie_title: title, rating });
    user.ratings = res.ratings; localStorage.setItem('cm_user', JSON.stringify(user));
    // Update stars in modal
    const row = document.getElementById(`star-row-${id}`);
    if (row) row.querySelectorAll('.star').forEach(s => s.classList.toggle('lit', parseInt(s.dataset.val) <= rating));
    toast(`Rated ${rating}★`, 'ok');
  } catch (e) { toast('Rating failed', 'err'); }
};

// ── CARD BUILDERS ─────────────────────────────
function tmdbCard(m, i = 0) {
  const poster = m.poster_path || ph(175, 263);
  const rating = m.vote_average?.toFixed(1) || 'N/A';
  const d = Math.min(i * 0.04, 0.5);
  const reason = m.reason || '';
  return `
    <div class="mc" style="animation-delay:${d}s" onclick="openModal(${m.id || 0},'${esc(m.title)}')">
      <img class="mc-poster" src="${poster}" loading="lazy" alt="${m.title}">
      ${reason ? `<div class="mc-reason-tag">${reason}</div>` : ''}
      <div class="mc-overlay">
        <div class="mc-name">${m.title}</div>
        <div class="mc-meta-row"><span class="mc-rating">★${rating}</span><span class="mc-year">${m.release_date?.slice(0, 4) || ''}</span></div>
      </div>
      <div class="mc-actions">
        <button class="mc-btn ${user?.liked_movies?.includes(m.title) ? 'liked' : ''}" onclick="event.stopPropagation();toggleLike('${esc(m.title)}')" title="Like">♥</button>
        <button class="mc-btn ${user?.watchlist?.includes(m.title) ? 'in-watchlist' : ''}" onclick="event.stopPropagation();addToWatchlist('${esc(m.title)}')" title="Watchlist">🎯</button>
      </div>
      <div class="mc-bottom"><div class="mc-bottom-title">${m.title}</div></div>
    </div>`;
}

function localCard(m, i = 0, grid = false) {
  const d = Math.min(i * 0.04, 0.5);
  // Using a transparent base64 image or a sleek low-contrast placeholder for immediate render
  const placeholder = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
  const displayImage = m.poster_path || placeholder;
  return `
    <div class="mc" style="${grid ? '' : 'flex:0 0 175px;'} animation-delay:${d}s" onclick="openModal(0,'${esc(m.title)}')">
      <div class="mc-poster-wrap" style="position:relative; width:100%; aspect-ratio:2/3; background:var(--bg3); display:flex; align-items:center; justify-content:center; overflow:hidden;">
        <div class="spinner" style="position:absolute; transform:scale(0.5); opacity:0.3; z-index:1;"></div>
        <img class="mc-poster" src="${displayImage}" loading="lazy" alt="${m.title}" id="lc-${i}-${m.title.slice(0, 5).replace(/[\s'"]/g, '_')}" style="position:relative; z-index:2; transition: opacity 0.3s ease; opacity: 0;" onload="this.style.opacity=1">
      </div>
      <div class="mc-overlay">
        <div class="mc-name">${m.title}</div>
        <div class="mc-meta-row"><span class="mc-rating">★${m.rating?.toFixed(1) || 'N/A'}</span></div>
      </div>
      <div class="mc-actions">
        <button class="mc-btn" onclick="event.stopPropagation();toggleLike('${esc(m.title)}')" title="Like">♥</button>
        <button class="mc-btn" onclick="event.stopPropagation();addToWatchlist('${esc(m.title)}')" title="Watchlist">🎯</button>
      </div>
      <div class="mc-bottom"><div class="mc-bottom-title">${m.title}</div></div>
    </div>`;
}

// Batch-load posters: one API call for all titles, update DOM as they arrive
async function batchLoadPosters(titles, container) {
  if (!titles?.length) return;
  try {
    const batchMap = await api('/movies/batch-search', 'POST', { titles });
    titles.forEach((title, i) => {
      const safeId = `lc-${i}-${title.slice(0, 5).replace(/[\s'"]/g, '_')}`;
      const img = container.querySelector(`#${safeId}`);
      const data = batchMap[title];
      if (img && data?.poster_path) img.src = data.poster_path;
    });
  } catch (e) { /* silently skip poster errors */ }
}

// ── UTILITIES ─────────────────────────────────
async function api(endpoint, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(`${API}${endpoint}`, opts);
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || 'API Error'); }
  return res.json();
}

function esc(s = '') { return String(s).replace(/'/g, "\\'").replace(/"/g, '&quot;'); }
function ph(w, h) { return `https://via.placeholder.com/${w}x${h}/16162a/505070?text=...`; }
function skelGrid(n) { return Array(n).fill('<div class="skel" style="aspect-ratio:2/3;border-radius:16px"></div>').join(''); }

function toast(msg, type = 'ok') {
  const c = document.getElementById('toasts');
  const t = document.createElement('div');
  t.className = `toast ${type}`; t.textContent = msg;
  c.appendChild(t); setTimeout(() => t.remove(), 3500);
}

// ── AI CHATBOT ─────────────────────────────────────
let chatOpen = false;
let chatHistory = [];
let chatIndustry = 'all';

window.toggleChat = function () {
  chatOpen = !chatOpen;
  const panel = document.getElementById('chat-panel');
  const bubble = document.getElementById('chat-bubble');
  if (chatOpen) {
    panel.classList.remove('hidden');
    panel.classList.add('chat-panel-open');
    bubble.classList.add('bubble-active');
    document.getElementById('chat-input')?.focus();
    chatIndustry = cinemaMode === 'indian' ? (industry || 'all_indian') : 'all';
  } else {
    panel.classList.remove('chat-panel-open');
    panel.classList.add('hidden');
    bubble.classList.remove('bubble-active');
  }
};

window.sendChatSuggestion = function (text) {
  const input = document.getElementById('chat-input');
  if (input) { input.value = text; sendChatMessage(); }
};

window.sendChatMessage = async function () {
  const input = document.getElementById('chat-input');
  const msg = input?.value?.trim();
  if (!msg) return;
  input.value = '';

  // Hide suggestions after first message
  const sugg = document.getElementById('chat-suggestions');
  if (sugg) sugg.style.display = 'none';

  // Append user message
  appendChatMsg(msg, 'user');
  chatHistory.push({ role: 'user', content: msg });

  // Show typing indicator
  const typingId = 'chat-typing-' + Date.now();
  const msgs = document.getElementById('chat-messages');
  msgs.insertAdjacentHTML('beforeend', `<div id="${typingId}" class="chat-msg bot"><div class="chat-typing"><span></span><span></span><span></span></div></div>`);
  msgs.scrollTop = msgs.scrollHeight;

  // Disable send button
  const btn = document.getElementById('chat-send-btn');
  if (btn) btn.disabled = true;

  try {
    const res = await api('/chat', 'POST', {
      message: msg,
      username: user?.username || '',
      industry: chatIndustry,
      history: chatHistory.slice(-8)
    });

    // Remove typing indicator
    document.getElementById(typingId)?.remove();

    const reply = res.reply || 'Here are some great picks for you!';
    appendChatMsg(reply, 'bot');
    chatHistory.push({ role: 'assistant', content: reply });

    // Render movie cards inline in chat
    if (res.movies?.length) {
      const movieHtml = `<div class="chat-movies">${res.movies.slice(0, 6).map(m => chatMovieCard(m)).join('')}</div>`;
      const msgWrap = document.createElement('div');
      msgWrap.className = 'chat-msg bot chat-movies-wrap';
      msgWrap.innerHTML = movieHtml;
      document.getElementById('chat-messages').appendChild(msgWrap);
      msgs.scrollTop = msgs.scrollHeight;
    }
  } catch (e) {
    document.getElementById(typingId)?.remove();
    appendChatMsg('Sorry, I had trouble connecting. Please try again! 🙏', 'bot');
  } finally {
    if (btn) btn.disabled = false;
    msgs.scrollTop = msgs.scrollHeight;
  }
};

function appendChatMsg(text, role) {
  const msgs = document.getElementById('chat-messages');
  if (!msgs) return;
  const d = document.createElement('div');
  d.className = `chat-msg ${role}`;
  d.innerHTML = `<div class="chat-bubble-msg">${text}</div>`;
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
}

function chatMovieCard(m) {
  const poster = m.poster_path || m.link || ph(100, 150);
  const title = m.title || '';
  const rating = m.vote_average ? `★ ${m.vote_average.toFixed(1)}` : (m.rating ? `★ ${parseFloat(m.rating).toFixed(1)}` : '');
  const id = m.id || 0;
  return `
    <div class="chat-mc" onclick="openModal(${id},'${esc(title)}')">
      <img src="${poster}" alt="${esc(title)}" onerror="this.src='${ph(100, 150)}'">
      <div class="chat-mc-title">${title}</div>
      ${rating ? `<div class="chat-mc-rating">${rating}</div>` : ''}
    </div>`;
}

// Enter key sends message
document.getElementById('chat-input')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') sendChatMessage();
});
// Defer binding since element may be in static HTML
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('chat-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') sendChatMessage();
  });
});
