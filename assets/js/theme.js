(function () {
    const root = document.documentElement;
    const key = "theme";
  
    // Set initial theme: saved preference > OS preference
    const saved = localStorage.getItem(key);
    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  
    if (saved === "dark" || (saved === null && prefersDark)) {
      root.setAttribute("data-theme", "dark");
    }
  
    function updateIcon() {
      const btn = document.querySelector(".theme-toggle");
      if (!btn) return;
      const isDark = root.getAttribute("data-theme") === "dark";
      btn.querySelector(".theme-toggle-icon").textContent = isDark ? "☀️" : "🌙";
      btn.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
    }
  
    document.addEventListener("click", (e) => {
      const btn = e.target.closest(".theme-toggle");
      if (!btn) return;
  
      const isDark = root.getAttribute("data-theme") === "dark";
      if (isDark) {
        root.removeAttribute("data-theme");
        localStorage.setItem(key, "light");
      } else {
        root.setAttribute("data-theme", "dark");
        localStorage.setItem(key, "dark");
      }
      updateIcon();
    });
  
    document.addEventListener("DOMContentLoaded", updateIcon);
    updateIcon();
  })();
  