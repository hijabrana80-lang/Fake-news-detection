/* TRUTHLINE — fact-check workspace.
   Character counter, sample loader and the staged analysis animation.

   The sequence is presentation only: it runs while a real /api/predict
   request is in flight and never resolves to a result the backend did
   not return. If the request fails the overlay closes and the real
   error is shown — no stage is ever marked complete on a failure. */
(function () {
  "use strict";

  var MIN_CHARS = 40;
  var STAGE_MS = 240;
  var MAX_SOFT_LIMIT = 18000;

  var form = document.getElementById("analyze-form");
  var textarea = document.getElementById("article-text");
  var headline = document.getElementById("headline");
  var counter = document.getElementById("char-count");
  var clearBtn = document.getElementById("clear-btn");
  var sampleBtn = document.getElementById("sample-btn");
  var submitBtn = document.getElementById("submit-btn");
  var overlay = document.getElementById("scan-overlay");
  var progressBar = document.getElementById("scan-progress-bar");
  var errorBox = document.getElementById("form-error");
  var stages = overlay ? overlay.querySelectorAll("[data-stage]") : [];

  var SAMPLES = [
    {
      title: "Senate passes bipartisan infrastructure bill",
      text:
        "WASHINGTON (Reuters) - The Senate on Tuesday passed a bipartisan infrastructure bill " +
        "by a vote of 69-30, sending the legislation to the House for consideration. The " +
        "package allocates $550 billion in new federal spending over five years for roads, " +
        "bridges, passenger rail and broadband expansion, according to the Congressional " +
        "Budget Office. Senators from both parties said the vote reflected a rare moment of " +
        "cooperation on fiscal policy."
    },
    {
      title: "",
      text:
        "BREAKING!!! You WON'T BELIEVE what this celebrity did next!! Doctors HATE this one " +
        "weird trick that cures diabetes instantly. Share before this is BANNED by the " +
        "mainstream media - the truth they don't want you to know. Thousands of insiders have " +
        "confirmed the conspiracy in a secret report that was censored overnight."
    }
  ];
  var sampleIndex = 0;

  if (!form || !textarea) return;

  /* ---------------------------------------------------------------- */
  /* Character counter                                                 */
  /* ---------------------------------------------------------------- */
  function updateCounter() {
    if (!counter) return;
    var length = textarea.value.length;
    counter.textContent = length.toLocaleString() + " characters";
    counter.classList.toggle("is-near", length > MAX_SOFT_LIMIT);
  }

  textarea.addEventListener("input", function () {
    clearError();
    updateCounter();
  });
  updateCounter();

  /* ---------------------------------------------------------------- */
  /* Inline error                                                      */
  /* ---------------------------------------------------------------- */
  function showError(message) {
    if (!errorBox) return;
    errorBox.textContent = message;
    errorBox.classList.add("is-visible");
  }

  function clearError() {
    if (!errorBox) return;
    errorBox.textContent = "";
    errorBox.classList.remove("is-visible");
  }

  /* ---------------------------------------------------------------- */
  /* Clear and sample buttons                                          */
  /* ---------------------------------------------------------------- */
  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      textarea.value = "";
      if (headline) headline.value = "";
      clearError();
      updateCounter();
      textarea.focus();
    });
  }

  if (sampleBtn) {
    sampleBtn.addEventListener("click", function () {
      var sample = SAMPLES[sampleIndex % SAMPLES.length];
      sampleIndex += 1;
      textarea.value = sample.text;
      if (headline) headline.value = sample.title;
      clearError();
      updateCounter();
      textarea.focus();
      textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    });
  }

  /* ---------------------------------------------------------------- */
  /* Staged analysis                                                   */
  /* ---------------------------------------------------------------- */
  var cancelled = false;

  function resetStages() {
    Array.prototype.forEach.call(stages, function (stage, index) {
      stage.classList.remove("is-active", "is-done");
      stage.querySelector(".stage-mark").textContent = String(index + 1);
    });
    if (progressBar) progressBar.style.width = "0%";
  }

  function animateStages() {
    return new Promise(function (resolve) {
      var index = 0;

      function step() {
        if (cancelled) return;

        if (index > 0) {
          var previous = stages[index - 1];
          previous.classList.remove("is-active");
          previous.classList.add("is-done");
          previous.querySelector(".stage-mark").textContent = "✓";
        }

        if (index >= stages.length) {
          if (progressBar) progressBar.style.width = "100%";
          resolve();
          return;
        }

        stages[index].classList.add("is-active");
        if (progressBar) {
          progressBar.style.width = Math.round(((index + 1) / stages.length) * 100) + "%";
        }
        index += 1;
        window.setTimeout(step, STAGE_MS);
      }

      step();
    });
  }

  function finishWithError(message) {
    cancelled = true;
    if (overlay) {
      overlay.classList.remove("is-active");
      overlay.setAttribute("aria-busy", "false");
    }
    if (submitBtn) submitBtn.disabled = false;
    resetStages();
    showError(message || "Analysis failed. Please try again.");
  }

  function runAnalysis(text, title) {
    cancelled = false;
    resetStages();
    clearError();

    if (overlay) {
      overlay.classList.add("is-active");
      overlay.setAttribute("aria-busy", "true");
    }
    if (submitBtn) submitBtn.disabled = true;

    var stagesDone = animateStages();

    var request = window
      .fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, title: title })
      })
      .then(function (response) {
        return response
          .json()
          .catch(function () {
            return {};
          })
          .then(function (data) {
            return { ok: response.ok, data: data };
          });
      });

    Promise.all([request, stagesDone])
      .then(function (results) {
        var response = results[0];
        if (!response.ok || !response.data || !response.data.record) {
          throw new Error(
            (response.data && response.data.error) || "The analyzer could not process this text."
          );
        }
        window.location.href = "/result/" + response.data.record.id;
      })
      .catch(function (error) {
        finishWithError(error && error.message);
      });
  }

  /* ---------------------------------------------------------------- */
  /* Submit                                                            */
  /* ---------------------------------------------------------------- */
  form.addEventListener("submit", function (event) {
    var text = textarea.value.trim();

    if (text.length < MIN_CHARS) {
      event.preventDefault();
      showError(
        "Article text is too short — at least " + MIN_CHARS + " characters are required."
      );
      textarea.focus();
      return;
    }

    /* Without fetch there is no animation: fall through to the native
       form POST so the workflow still completes. */
    if (typeof window.fetch !== "function") return;

    event.preventDefault();
    runAnalysis(text, headline ? headline.value.trim() : "");
  });
})();
