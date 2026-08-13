(function () {
  "use strict";

  var supported = ["python", "r", "duckdb"];
  var selects = Array.prototype.slice.call(
    document.querySelectorAll("[data-language-select]")
  );
  var panels = Array.prototype.slice.call(
    document.querySelectorAll("[data-language-panel]")
  );

  if (!selects.length || !panels.length) return;

  var isSupported = function (value) {
    return supported.indexOf(value) !== -1;
  };

  var storedLanguage = null;
  try {
    storedLanguage = window.localStorage.getItem("stride-align-language");
  } catch (_error) {
    storedLanguage = null;
  }

  var requestedLanguage = new URLSearchParams(window.location.search).get("language");
  var initialLanguage = isSupported(requestedLanguage)
    ? requestedLanguage
    : (isSupported(storedLanguage) ? storedLanguage : selects[0].value);

  var showLanguage = function (language, updateAddress) {
    if (!isSupported(language)) return;

    selects.forEach(function (select) {
      select.value = language;
    });
    panels.forEach(function (panel) {
      panel.hidden = panel.getAttribute("data-language-panel") !== language;
    });
    document.body.setAttribute("data-language", language);

    try {
      window.localStorage.setItem("stride-align-language", language);
    } catch (_error) {
      // Selection still synchronizes in privacy-restricted contexts.
    }

    if (updateAddress && window.history && window.history.replaceState) {
      var url = new URL(window.location.href);
      url.searchParams.set("language", language);
      window.history.replaceState(null, "", url.toString());
    }
  };

  selects.forEach(function (select) {
    select.addEventListener("change", function () {
      showLanguage(select.value, true);
    });
  });

  showLanguage(initialLanguage, false);
})();
