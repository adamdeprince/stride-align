(function () {
  "use strict";

  var supported = ["python", "r", "duckdb", "postgres", "memgraph"];
  var selects = Array.prototype.slice.call(
    document.querySelectorAll("[data-language-select]")
  );
  var panels = Array.prototype.slice.call(
    document.querySelectorAll("[data-language-panel]")
  );
  var pickers = [];

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

  var updatePicker = function (picker, language) {
    var selected = picker.options.filter(function (option) {
      return option.getAttribute("data-language-value") === language;
    })[0];

    if (selected) picker.value.textContent = selected.textContent;
    picker.options.forEach(function (option) {
      option.setAttribute(
        "aria-selected",
        option.getAttribute("data-language-value") === language ? "true" : "false"
      );
    });
  };

  var showLanguage = function (language, updateAddress) {
    if (!isSupported(language)) return;

    selects.forEach(function (select) {
      select.value = language;
    });
    pickers.forEach(function (picker) {
      updatePicker(picker, language);
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

  var closePicker = function (picker, restoreFocus) {
    picker.menu.hidden = true;
    picker.trigger.setAttribute("aria-expanded", "false");
    if (restoreFocus) picker.trigger.focus();
  };

  var closeOtherPickers = function (current) {
    pickers.forEach(function (picker) {
      if (picker !== current) closePicker(picker, false);
    });
  };

  var selectedIndex = function (picker) {
    var index = picker.options.findIndex(function (option) {
      return option.getAttribute("aria-selected") === "true";
    });
    return index < 0 ? 0 : index;
  };

  var focusOption = function (picker, index) {
    var count = picker.options.length;
    if (!count) return;
    picker.options[(index + count) % count].focus();
  };

  var openPicker = function (picker, index) {
    closeOtherPickers(picker);
    picker.menu.hidden = false;
    picker.trigger.setAttribute("aria-expanded", "true");
    focusOption(picker, typeof index === "number" ? index : selectedIndex(picker));
  };

  var chooseOption = function (picker, option) {
    showLanguage(option.getAttribute("data-language-value"), true);
    closePicker(picker, true);
  };

  var enhanceSelect = function (select) {
    var wrapper = select.parentElement;
    var control = wrapper && wrapper.parentElement;
    if (!wrapper || !control || !wrapper.classList.contains("example-language-picker")) {
      return null;
    }

    var label = control.querySelector("label");
    var trigger = document.createElement("button");
    var value = document.createElement("span");
    var chevron = document.createElement("span");
    var menu = document.createElement("div");
    var triggerId = select.id + "-trigger";
    var menuId = select.id + "-menu";

    if (label) {
      label.id = select.id + "-label";
      label.removeAttribute("for");
    }

    trigger.type = "button";
    trigger.id = triggerId;
    trigger.className = "example-language-trigger";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    trigger.setAttribute("aria-controls", menuId);

    value.id = triggerId + "-value";
    value.className = "example-language-value";
    if (label) trigger.setAttribute("aria-labelledby", label.id + " " + value.id);
    chevron.className = "example-language-chevron";
    chevron.setAttribute("aria-hidden", "true");
    chevron.textContent = "⌄";
    trigger.appendChild(value);
    trigger.appendChild(chevron);

    menu.id = menuId;
    menu.className = "example-language-menu";
    menu.setAttribute("role", "listbox");
    if (label) menu.setAttribute("aria-labelledby", label.id);
    menu.hidden = true;

    var optionNodes = Array.prototype.map.call(select.options, function (nativeOption) {
      var option = document.createElement("button");
      option.type = "button";
      option.className = "example-language-option";
      option.setAttribute("role", "option");
      option.setAttribute("data-language-value", nativeOption.value);
      option.textContent = nativeOption.textContent;
      menu.appendChild(option);
      return option;
    });

    wrapper.appendChild(trigger);
    wrapper.appendChild(menu);
    wrapper.classList.add("is-enhanced");
    select.tabIndex = -1;
    select.setAttribute("aria-hidden", "true");

    var picker = {
      wrapper: wrapper,
      trigger: trigger,
      value: value,
      menu: menu,
      options: optionNodes
    };

    trigger.addEventListener("click", function () {
      if (menu.hidden) {
        openPicker(picker);
      } else {
        closePicker(picker, false);
      }
    });

    trigger.addEventListener("keydown", function (event) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        openPicker(picker);
      } else if (event.key === "Home") {
        event.preventDefault();
        openPicker(picker, 0);
      } else if (event.key === "End") {
        event.preventDefault();
        openPicker(picker, optionNodes.length - 1);
      } else if (event.key === "Escape") {
        closePicker(picker, false);
      }
    });

    optionNodes.forEach(function (option, index) {
      option.addEventListener("click", function () {
        chooseOption(picker, option);
      });
      option.addEventListener("keydown", function (event) {
        if (event.key === "ArrowDown") {
          event.preventDefault();
          focusOption(picker, index + 1);
        } else if (event.key === "ArrowUp") {
          event.preventDefault();
          focusOption(picker, index - 1);
        } else if (event.key === "Home") {
          event.preventDefault();
          focusOption(picker, 0);
        } else if (event.key === "End") {
          event.preventDefault();
          focusOption(picker, optionNodes.length - 1);
        } else if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          chooseOption(picker, option);
        } else if (event.key === "Escape") {
          event.preventDefault();
          closePicker(picker, true);
        } else if (event.key === "Tab") {
          closePicker(picker, false);
        }
      });
    });

    return picker;
  };

  selects.forEach(function (select) {
    select.addEventListener("change", function () {
      showLanguage(select.value, true);
    });
    var picker = enhanceSelect(select);
    if (picker) pickers.push(picker);
  });

  document.addEventListener("click", function (event) {
    pickers.forEach(function (picker) {
      if (!picker.wrapper.contains(event.target)) closePicker(picker, false);
    });
  });

  showLanguage(initialLanguage, false);
})();
