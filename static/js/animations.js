/* TRUTHLINE — motion controller.
   Everything here is a progressive enhancement: without JavaScript (or
   with prefers-reduced-motion) every value, chart and chain node is
   already rendered in its final state by the server.

   The file only does four things:
     1. triggers class-driven sequences when they enter the viewport
     2. counts numbers up from 0 to the real server value
     3. grows donut / confidence rings from zero to the real value
     4. a very small scroll parallax on one decorative element          */
(function () {
  "use strict";

  var reduceQuery = window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)")
    : null;
  var reduce = reduceQuery ? reduceQuery.matches : false;
  var hasIO = "IntersectionObserver" in window;

  function each(list, fn) {
    Array.prototype.forEach.call(list, fn);
  }

  /* ---------------------------------------------------------------- */
  /* Viewport triggers                                                 */
  /* ---------------------------------------------------------------- */
  var observer = null;

  if (hasIO && !reduce) {
    observer = new IntersectionObserver(
      function (entries, obs) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          obs.unobserve(entry.target);
          var run = entry.target.__onEnter;
          if (typeof run !== "function") return;
          /* One decorative effect must never break the others. */
          try {
            run.call(entry.target);
          } catch (error) {
            if (window.console && window.console.warn) {
              window.console.warn("TRUTHLINE motion skipped:", error);
            }
          }
        });
      },
      { rootMargin: "0px 0px -15% 0px", threshold: 0 }
    );
  }

  function onEnter(el, fn) {
    if (!el || reduce) return;
    if (!observer) {
      fn.call(el);
      return;
    }
    el.__onEnter = fn;
    observer.observe(el);
  }

  function activate(selector, className) {
    each(document.querySelectorAll(selector), function (el) {
      onEnter(el, function () {
        el.classList.add(className);
      });
    });
  }

  activate(".chain-visual", "is-built");
  activate(".balance", "is-settled");
  activate(".table-wrap", "is-in");
  activate(".colophon", "is-in");

  /* ---------------------------------------------------------------- */
  /* Count-up: 0 → the value the server already rendered               */
  /* ---------------------------------------------------------------- */
  function countUp(el, duration) {
    var raw = (el.getAttribute("data-count") || el.textContent || "").trim();
    var match = raw.match(/(-?[\d][\d,]*(?:\.\d+)?)/);
    if (!match || typeof match[1] !== "string") return;

    var token = match[1];
    var target = parseFloat(token.replace(/,/g, ""));
    if (isNaN(target)) return;

    var decimals = (token.split(".")[1] || "").length;
    var prefix = raw.slice(0, match.index);
    var suffix = raw.slice(match.index + token.length);
    var start = null;

    function frame(now) {
      if (start === null) start = now;
      var progress = Math.min(1, (now - start) / duration);
      var eased = 1 - Math.pow(1 - progress, 3);
      el.textContent =
        prefix +
        (target * eased).toLocaleString(undefined, {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals
        }) +
        suffix;
      if (progress < 1) window.requestAnimationFrame(frame);
    }

    window.requestAnimationFrame(frame);
  }

  each(document.querySelectorAll(".stat-value"), function (el) {
    onEnter(el, function () {
      countUp(el, 950);
    });
  });

  each(document.querySelectorAll(".score-value"), function (el) {
    onEnter(el, function () {
      countUp(el, 1100);
    });
  });

  /* ---------------------------------------------------------------- */
  /* Charts: donut segments and the confidence ring                    */
  /* ---------------------------------------------------------------- */
  function growFromZero(circle) {
    var finalValue = circle.getAttribute("stroke-dasharray");
    if (!finalValue || reduce || !circle.style) return;

    circle.style.transition = "none";
    circle.style.strokeDasharray = "0 326.73";

    onEnter(circle, function () {
      circle.style.transition = "";
      circle.style.strokeDasharray = finalValue;
    });
  }

  each(document.querySelectorAll(".donut-real, .donut-fake"), growFromZero);
  each(document.querySelectorAll(".score-ring-fill"), growFromZero);

  /* ---------------------------------------------------------------- */
  /* Subtle parallax on the hero illustration                          */
  /* ---------------------------------------------------------------- */
  var parallax = document.querySelectorAll("[data-parallax]");

  if (parallax.length && !reduce && window.innerWidth >= 768) {
    var ticking = false;

    var update = function () {
      ticking = false;
      var y = window.pageYOffset || document.documentElement.scrollTop || 0;
      each(parallax, function (el) {
        var speed = parseFloat(el.getAttribute("data-parallax")) || 0.05;
        var shift = y * speed;
        if (shift > 22) shift = 22;
        if (shift < -22) shift = -22;
        el.style.setProperty("--shift", shift.toFixed(2) + "px");
      });
    };

    window.addEventListener(
      "scroll",
      function () {
        if (ticking) return;
        ticking = true;
        window.requestAnimationFrame(update);
      },
      { passive: true }
    );

    update();
  }

  /* ---------------------------------------------------------------- */
  /* Verification form: loading state while the request is in flight   */
  /* ---------------------------------------------------------------- */
  var verifyForm = document.querySelector(".verify-form");

  if (verifyForm) {
    verifyForm.addEventListener("submit", function () {
      var button = verifyForm.querySelector("button[type='submit']");
      if (!button) return;
      button.classList.add("is-loading");
      button.setAttribute("aria-busy", "true");
      window.setTimeout(function () {
        button.classList.remove("is-loading");
        button.removeAttribute("aria-busy");
      }, 6000);
    });
  }
})();
