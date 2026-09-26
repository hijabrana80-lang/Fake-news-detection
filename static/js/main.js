/* TRUTHLINE — global progressive enhancement.
   Navigation, scroll reveals and keyboard shortcuts. Page specific logic
   lives in its own file (see analyze.js). */
(function () {
  "use strict";

  /* ---------------------------------------------------------------- */
  /* Mobile navigation                                                  */
  /* ---------------------------------------------------------------- */
  var toggle = document.querySelector(".nav-toggle");
  var nav = document.getElementById("primary-nav");

  function setNav(open) {
    if (!toggle || !nav) return;
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Close navigation menu" : "Open navigation menu");
    nav.classList.toggle("is-open", open);
  }

  if (toggle && nav) {
    toggle.addEventListener("click", function () {
      setNav(toggle.getAttribute("aria-expanded") !== "true");
    });

    nav.addEventListener("click", function (event) {
      if (event.target.closest("a")) setNav(false);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") setNav(false);
    });

    window.addEventListener("resize", function () {
      if (window.innerWidth > 880) setNav(false);
    });
  }

  /* ---------------------------------------------------------------- */
  /* Reveal sections as they enter the viewport                        */
  /* ---------------------------------------------------------------- */
  var revealables = document.querySelectorAll(".reveal");

  if (!revealables.length) return;

  if (!("IntersectionObserver" in window)) {
    Array.prototype.forEach.call(revealables, function (el) {
      el.classList.add("is-visible");
    });
    return;
  }

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
  );

  Array.prototype.forEach.call(revealables, function (el) {
    observer.observe(el);
  });
})();
