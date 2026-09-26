/* Progressive enhancement for the detection form. */
(function () {
  "use strict";

  var textarea = document.getElementById("article-text");
  var counter = document.getElementById("char-count");
  var form = document.getElementById("check-form");
  var button = document.getElementById("submit-btn");
  var sampleBtn = document.getElementById("sample-btn");

  var SAMPLES = [
    "BREAKING!!! You WON'T BELIEVE what this celebrity did next!! Doctors HATE this one weird trick that cures diabetes instantly. Share before this is BANNED by the mainstream media — the truth they don't want you to know. Thousands of insiders have confirmed the conspiracy in a secret report that was censored overnight.",
    "WASHINGTON (Reuters) - The Senate on Tuesday passed a bipartisan infrastructure bill by a vote of 69-30, sending the legislation to the House for consideration. The package allocates $550 billion in new federal spending over five years for roads, bridges, passenger rail and broadband expansion, according to the Congressional Budget Office."
  ];
  var sampleIndex = 0;

  if (textarea && counter) {
    var update = function () {
      counter.textContent = textarea.value.length.toLocaleString();
    };
    textarea.addEventListener("input", update);
    update();
  }

  if (sampleBtn && textarea) {
    sampleBtn.addEventListener("click", function () {
      textarea.value = SAMPLES[sampleIndex % SAMPLES.length];
      sampleIndex += 1;
      textarea.dispatchEvent(new Event("input"));
      textarea.focus();
    });
  }

  if (form && button) {
    form.addEventListener("submit", function () {
      button.disabled = true;
      button.textContent = "Analysing…";
    });
  }
})();
