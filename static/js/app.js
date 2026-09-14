/**
 * MedAI Frontend Application
 * Handles: Chat, Disease Prediction, X-Ray Analysis, Patient PDF Report
 */

"use strict";

// ─────────────────────────────────────────────
// Tab Navigation
// ─────────────────────────────────────────────

document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const tabId = btn.dataset.tab;

    document
      .querySelectorAll(".nav-btn")
      .forEach(b => b.classList.remove("active"));

    document
      .querySelectorAll(".tab")
      .forEach(t => t.classList.remove("active"));

    btn.classList.add("active");

    const target = document.getElementById(`tab-${tabId}`);

    if (target) {
      target.classList.add("active");
    }
  });
});

// ─────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────

function showLoading(text = "Processing...") {
  const loadingText =
    document.getElementById("loading-text");

  const loadingOverlay =
    document.getElementById("loading-overlay");

  if (loadingText) {
    loadingText.textContent = text;
  }

  if (loadingOverlay) {
    loadingOverlay.style.display = "flex";
  }
}

function hideLoading() {
  const loadingOverlay =
    document.getElementById("loading-overlay");

  if (loadingOverlay) {
    loadingOverlay.style.display = "none";
  }
}

function showToast(message, type = "error") {
  const existing =
    document.querySelector(".toast");

  if (existing) {
    existing.remove();
  }

  const toast =
    document.createElement("div");

  toast.className = `toast ${type}`;
  toast.textContent = message;

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 4000);
}

function autoResize(el) {
  if (!el) {
    return;
  }

  el.style.height = "auto";

  el.style.height =
    Math.min(el.scrollHeight, 120) + "px";
}

function handleKey(e) {
  if (
    e.key === "Enter" &&
    !e.shiftKey
  ) {
    e.preventDefault();
    sendMessage();
  }
}

// ─────────────────────────────────────────────
// Chat
// ─────────────────────────────────────────────

function fillChat(text) {
  const input =
    document.getElementById("chat-input");

  if (!input) {
    return;
  }

  input.value = text;

  autoResize(input);

  input.focus();
}

function clearChat() {
  const chatWindow =
    document.getElementById("chat-window");

  if (chatWindow) {
    chatWindow.innerHTML = `
      <div class="welcome-msg">
        <div class="welcome-icon">
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
            <circle
              cx="20"
              cy="20"
              r="19"
              stroke="var(--accent)"
              stroke-width="1.5"
            />

            <path
              d="M20 10v20M10 20h20"
              stroke="var(--accent)"
              stroke-width="2.5"
              stroke-linecap="round"
            />
          </svg>
        </div>

        <h2>Welcome to MedAI</h2>

        <p>
          Ask any medical question. I'll provide
          evidence-based answers from medical literature.
        </p>

        <p class="disclaimer-small">
          ⚠️ Not a substitute for professional medical advice.
        </p>
      </div>
    `;
  }

  fetch(
    "/api/chat/clear",
    {
      method: "POST"
    }
  ).catch(() => {});
}

function appendMessage(
  role,
  content,
  sources = [],
  disclaimer = ""
) {
  const chatWindow =
    document.getElementById("chat-window");

  if (!chatWindow) {
    return;
  }

  const welcome =
    chatWindow.querySelector(
      ".welcome-msg"
    );

  if (welcome) {
    welcome.remove();
  }

  const msgDiv =
    document.createElement("div");

  msgDiv.className =
    `message ${role}`;

  const avatarText =
    role === "user"
      ? "👤"
      : "🩺";

  const sourcesHTML =
    Array.isArray(sources) &&
    sources.length > 0
      ? `
        <div class="message-sources">
          <div class="sources-title">
            Sources
          </div>

          ${sources
            .map(
              source =>
                `<div class="source-item">${escapeHTML(
                  String(source)
                )}</div>`
            )
            .join("")}
        </div>
      `
      : "";

  const disclaimerHTML =
    disclaimer
      ? `
        <div class="message-disclaimer">
          ${escapeHTML(String(disclaimer))}
        </div>
      `
      : "";

  msgDiv.innerHTML = `
    <div class="message-avatar">
      ${avatarText}
    </div>

    <div class="message-bubble">
      ${escapeHTML(
        String(content || "")
      ).replace(/\n/g, "<br>")}

      ${sourcesHTML}

      ${disclaimerHTML}
    </div>
  `;

  chatWindow.appendChild(msgDiv);

  chatWindow.scrollTop =
    chatWindow.scrollHeight;

  return msgDiv;
}

function escapeHTML(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function showTyping() {
  const chatWindow =
    document.getElementById("chat-window");

  if (!chatWindow) {
    return;
  }

  const typingDiv =
    document.createElement("div");

  typingDiv.className =
    "message bot";

  typingDiv.id =
    "typing-indicator";

  typingDiv.innerHTML = `
    <div class="message-avatar">
      🩺
    </div>

    <div class="message-bubble">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;

  chatWindow.appendChild(
    typingDiv
  );

  chatWindow.scrollTop =
    chatWindow.scrollHeight;
}

function removeTyping() {
  const el =
    document.getElementById(
      "typing-indicator"
    );

  if (el) {
    el.remove();
  }
}

async function sendMessage() {
  const input =
    document.getElementById("chat-input");

  const sendBtn =
    document.getElementById("send-btn");

  if (!input) {
    return;
  }

  const message =
    input.value.trim();

  if (!message) {
    return;
  }

  input.value = "";
  input.style.height = "auto";

  if (sendBtn) {
    sendBtn.disabled = true;
  }

  appendMessage(
    "user",
    message
  );

  showTyping();

  try {
    const res =
      await fetch(
        "/api/chat",
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify({
            message
          })
        }
      );

    const data =
      await res.json();

    removeTyping();

    if (!res.ok) {
      appendMessage(
        "bot",
        `Error: ${
          data.error ||
          "Something went wrong."
        }`
      );

      return;
    }

    appendMessage(
      "bot",
      data.answer,
      data.sources,
      data.disclaimer
    );

  } catch (err) {
    console.error(
      "Chat error:",
      err
    );

    removeTyping();

    appendMessage(
      "bot",
      "⚠️ Unable to connect to MedAI server. Please check your connection."
    );

  } finally {
    if (sendBtn) {
      sendBtn.disabled = false;
    }

    input.focus();
  }
}

// ─────────────────────────────────────────────
// Disease Prediction
// ─────────────────────────────────────────────

// Store the latest assessment so it can be
// used to generate the PDF report.

let lastAssessmentData = null;
let lastAssessmentPredictions = null;


// IMPORTANT:
// These IDs match the actual fields in index.html.

function getFormData() {
  const fields = [
    "age",
    "sex",
    "BMI",
    "Pregnancies",
    "Glucose",
    "Insulin",
    "chol",
    "DiabetesPedigreeFunction",
    "trestbps",
    "thalach",
    "oldpeak",
    "ca",
    "SkinThickness",
    "cp",
    "exang",
    "fbs"
  ];

  const data = {};

  fields.forEach(fieldName => {
    const element =
      document.getElementById(
        fieldName
      );

    if (
      element &&
      element.value !== ""
    ) {
      const rawValue =
        element.value;

      const numericValue =
        Number(rawValue);

      if (
        rawValue !== "" &&
        !Number.isNaN(
          numericValue
        )
      ) {
        data[fieldName] =
          numericValue;
      } else {
        data[fieldName] =
          rawValue;
      }
    }
  });

  return data;
}

function getRiskClass(
  probability,
  prediction
) {
  const text =
    String(
      prediction || ""
    ).toLowerCase();

  const pct =
    Number(
      probability
    ) || 0;

  if (
    text.includes("low")
  ) {
    return "risk-low";
  }

  if (
    text.includes("high") ||
    pct > 60
  ) {
    return "risk-high";
  }

  return "risk-moderate";
}

function renderResults(
  predictions,
  disclaimer
) {
  const area =
    document.getElementById(
      "results-area"
    );

  if (!area) {
    return;
  }

  area.innerHTML = `
    <h2
      style="
        font-family:var(--font-display);
        font-size:20px;
        margin-bottom:20px
      "
    >
      Assessment Results
    </h2>
  `;

  if (
    !Array.isArray(predictions)
  ) {
    predictions = [];
  }

  predictions.forEach(
    pred => {
      const riskClass =
        getRiskClass(
          pred.probability,
          pred.prediction
        );

      const pct =
        Number(
          pred.probability
        ) || 0;

      const factorsHTML =
        (
          pred.contributing_factors ||
          []
        )
          .map(
            factor =>
              `<span class="result-factor-tag">${escapeHTML(
                String(factor)
              )}</span>`
          )
          .join("");

      const item =
        document.createElement(
          "div"
        );

      item.className =
        `result-item ${riskClass}`;

      item.innerHTML = `
        <div class="result-condition">
          ${escapeHTML(
            String(
              pred.condition ||
              "Assessment"
            )
          )}
        </div>

        <div class="result-prediction">
          ${escapeHTML(
            String(
              pred.prediction ||
              "N/A"
            )
          )}
        </div>

        <div class="result-bar-wrap">
          <div
            class="result-bar"
            style="width:0%"
            data-width="${pct}%"
          ></div>
        </div>

        <div class="result-pct">
          Risk Probability: ${pct}%
        </div>

        ${
          factorsHTML
            ? `
              <div class="result-factors">
                <div class="result-factors-title">
                  Contributing Factors
                </div>

                ${factorsHTML}
              </div>
            `
            : ""
        }
      `;

      area.appendChild(
        item
      );

      requestAnimationFrame(
        () => {
          const bar =
            item.querySelector(
              ".result-bar"
            );

          if (bar) {
            bar.style.width =
              `${pct}%`;
          }
        }
      );
    }
  );

  if (disclaimer) {
    const disc =
      document.createElement(
        "div"
      );

    disc.className =
      "results-disclaimer";

    disc.textContent =
      disclaimer;

    area.appendChild(
      disc
    );
  }

  // ─────────────────────────────────────
  // Download Medical Report Button
  // ─────────────────────────────────────

  const reportButton =
    document.createElement(
      "button"
    );

  reportButton.type =
    "button";

  reportButton.className =
    "primary-btn";

  reportButton.style.marginTop =
    "18px";

  reportButton.style.width =
    "100%";

  reportButton.style.padding =
    "12px 18px";

  reportButton.style.cursor =
    "pointer";

  reportButton.textContent =
    "📄 Download Medical Report";

  reportButton.addEventListener(
    "click",
    generatePatientReport
  );

  area.appendChild(
    reportButton
  );
}

async function runPrediction() {
  const data =
    getFormData();

  if (
    Object.keys(data).length === 0
  ) {
    showToast(
      "Please enter at least some patient data to run assessment.",
      "error"
    );

    return;
  }

  showLoading(
    "Running ML models..."
  );

  try {
    const res =
      await fetch(
        "/api/predict/disease",
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify(
            data
          )
        }
      );

    const result =
      await res.json();

    if (!res.ok) {
      showToast(
        result.error ||
          "Prediction failed.",
        "error"
      );

      return;
    }

    // Save data and predictions
    // for the PDF report.

    lastAssessmentData =
      data;

    lastAssessmentPredictions =
      result.predictions ||
      [];

    renderResults(
      result.predictions,
      result.disclaimer
    );

  } catch (err) {
    console.error(
      "Prediction error:",
      err
    );

    showToast(
      "Could not connect to prediction service.",
      "error"
    );

  } finally {
    hideLoading();
  }
}

// ─────────────────────────────────────────────
// Patient PDF Report
// ─────────────────────────────────────────────

async function generatePatientReport() {
  if (
    !lastAssessmentData ||
    !lastAssessmentPredictions
  ) {
    showToast(
      "Please run the patient assessment first.",
      "error"
    );

    return;
  }

  showLoading(
    "Generating medical report..."
  );

  try {
    const predictions =
      lastAssessmentPredictions;

    // Find diabetes result
    const diabetes =
      predictions.find(
        prediction =>
          prediction.condition &&
          prediction.condition
            .toLowerCase()
            .includes(
              "diabetes"
            )
      ) || {};

    // Find heart disease result
    const heart =
      predictions.find(
        prediction =>
          prediction.condition &&
          prediction.condition
            .toLowerCase()
            .includes(
              "heart"
            )
      ) || {};

    // Find cardiovascular result
    const cardiovascular =
      predictions.find(
        prediction =>
          prediction.condition &&
          (
            prediction.condition
              .toLowerCase()
              .includes(
                "cardiovascular"
              ) ||

            prediction.condition
              .toLowerCase()
              .includes(
                "risk score"
              )
          )
      ) || {};

    // Structure expected by
    // patient_report.py

    const reportResults = {
      diabetes: {
        prediction:
          diabetes.prediction ||
          "N/A",

        probability:
          diabetes.probability ??
          "N/A"
      },

      heart_disease: {
        prediction:
          heart.prediction ||
          "N/A",

        probability:
          heart.probability ??
          "N/A"
      },

      cardiovascular: {
        risk_level:
          cardiovascular.prediction ||
          "N/A",

        risk_percentage:
          cardiovascular.probability ??
          "N/A",

        factors:
          cardiovascular
            .contributing_factors ||
          []
      }
    };

    // Patient information

    const reportPayload = {
      patient: {
        name:
          lastAssessmentData.name ||
          "Not provided",

        age:
          lastAssessmentData.age ??
          "Not provided",

        sex:
          lastAssessmentData.sex ??
          "Not provided",

        bmi:
          lastAssessmentData.BMI ??
          lastAssessmentData.bmi ??
          "Not provided"
      },

      results:
        reportResults
    };

    const res =
      await fetch(
        "/api/generate-report",
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify(
            reportPayload
          )
        }
      );

    if (!res.ok) {
      let message =
        "Failed to generate medical report.";

      try {
        const errorData =
          await res.json();

        message =
          errorData.error ||
          message;

      } catch (_) {
        // Server did not return JSON.
      }

      showToast(
        message,
        "error"
      );

      return;
    }

    // Receive PDF binary data

    const blob =
      await res.blob();

    if (
      !blob ||
      blob.size === 0
    ) {
      showToast(
        "The generated report is empty.",
        "error"
      );

      return;
    }

    // Create browser download

    const url =
      window.URL.createObjectURL(
        blob
      );

    const link =
      document.createElement(
        "a"
      );

    link.href =
      url;

    link.download =
      "MedAI_Patient_Report.pdf";

    document.body.appendChild(
      link
    );

    link.click();

    link.remove();

    window.URL.revokeObjectURL(
      url
    );

    showToast(
      "Medical report downloaded successfully.",
      "success"
    );

  } catch (err) {
    console.error(
      "Patient report error:",
      err
    );

    showToast(
      "Could not generate the medical report.",
      "error"
    );

  } finally {
    hideLoading();
  }
}

// ─────────────────────────────────────────────
// X-Ray Analysis
// ─────────────────────────────────────────────

let xrayBase64 = null;

function handleDragOver(e) {
  e.preventDefault();

  const zone =
    document.getElementById(
      "upload-zone"
    );

  if (zone) {
    zone.classList.add(
      "dragover"
    );
  }
}

function handleDrop(e) {
  e.preventDefault();

  const zone =
    document.getElementById(
      "upload-zone"
    );

  if (zone) {
    zone.classList.remove(
      "dragover"
    );
  }

  const file =
    e.dataTransfer &&
    e.dataTransfer.files
      ? e.dataTransfer.files[0]
      : null;

  if (file) {
    processImageFile(
      file
    );
  }
}

function handleImageUpload(e) {
  const file =
    e.target &&
    e.target.files
      ? e.target.files[0]
      : null;

  if (file) {
    processImageFile(
      file
    );
  }
}

function processImageFile(file) {
  if (
    !file.type.startsWith(
      "image/"
    )
  ) {
    showToast(
      "Please upload a valid image file.",
      "error"
    );

    return;
  }

  if (
    file.size >
    10 * 1024 * 1024
  ) {
    showToast(
      "File size must be under 10MB.",
      "error"
    );

    return;
  }

  const reader =
    new FileReader();

  reader.onload =
    e => {
      const result =
        e.target.result;

      if (
        !result ||
        !result.startsWith(
          "data:image"
        )
      ) {
        showToast(
          "Invalid image file.",
          "error"
        );

        return;
      }

      xrayBase64 =
        result;

      const preview =
        document.getElementById(
          "preview-img"
        );

      const uploadZone =
        document.getElementById(
          "upload-zone"
        );

      const previewArea =
        document.getElementById(
          "xray-preview"
        );

      const results =
        document.getElementById(
          "xray-results"
        );

      if (preview) {
        preview.src =
          xrayBase64;
      }

      if (uploadZone) {
        uploadZone.style.display =
          "none";
      }

      if (previewArea) {
        previewArea.style.display =
          "block";
      }

      if (results) {
        results.style.display =
          "none";
      }
    };

  reader.readAsDataURL(
    file
  );
}

async function analyzeXray() {
  if (!xrayBase64) {
    showToast(
      "Please upload an X-ray image first.",
      "error"
    );

    return;
  }

  showLoading(
    "Analyzing X-ray with CNN..."
  );

  try {
    const res =
      await fetch(
        "/api/predict/xray",
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify({
            image:
              xrayBase64
          })
        }
      );

    const data =
      await res.json();

    if (
      !res.ok ||
      data.error
    ) {
      showToast(
        data.error ||
          "X-ray analysis failed.",
        "error"
      );

      return;
    }

    renderXrayResults(
      data
    );

  } catch (err) {
    console.error(
      "X-ray error:",
      err
    );

    showToast(
      "Could not connect to X-ray analysis service.",
      "error"
    );

  } finally {
    hideLoading();
  }
}

function renderXrayResults(data) {
  const resultsEl =
    document.getElementById(
      "xray-results"
    );

  if (!resultsEl) {
    return;
  }

  resultsEl.style.display =
    "block";

  if (
    !data ||
    !data.available
  ) {
    resultsEl.innerHTML = `
      <div
        class="xray-result-box"
        style="color:var(--text-muted)"
      >
        ${
          data &&
          data.error
            ? escapeHTML(
                data.error
              )
            : "X-ray analysis is currently unavailable."
        }
      </div>
    `;

    return;
  }

  const predClass =
    data.prediction ===
    "Normal"
      ? "normal"
      : "pneumonia";

  const probs =
    data.class_probabilities ||
    {};

  resultsEl.innerHTML = `
    <div class="xray-result-box">

      <div class="result-condition">
        X-Ray Classification
      </div>

      <div
        class="xray-prediction ${predClass}"
      >
        ${escapeHTML(
          String(
            data.prediction ||
            "N/A"
          )
        )}
      </div>

      <div class="result-pct">
        Confidence:
        ${data.confidence ?? "N/A"}%
      </div>

      <div
        class="xray-prob-bars"
        style="margin-top:16px"
      >

        ${Object.entries(
          probs
        )
          .map(
            ([cls, pct]) => `
              <div
                class="xray-prob-row"
              >

                <div
                  class="xray-prob-label"
                >
                  ${escapeHTML(
                    String(cls)
                  )}
                </div>

                <div
                  class="xray-prob-bar-wrap"
                >
                  <div
                    class="xray-prob-bar ${escapeHTML(
                      String(cls)
                        .toLowerCase()
                    )}"
                    style="width:${Number(
                      pct
                    ) || 0}%"
                  ></div>
                </div>

                <div
                  class="xray-prob-pct"
                >
                  ${pct}%
                </div>

              </div>
            `
          )
          .join("")}

      </div>

       

    </div>
  `;
}

// ─────────────────────────────────────────────
// Global Button / HTML Event Compatibility
// ─────────────────────────────────────────────

// Main assessment button
const predictButton =
  document.getElementById(
    "predict-btn"
  );

if (predictButton) {
  predictButton.addEventListener(
    "click",
    runPrediction
  );
}

// Some versions of the HTML may use
// these IDs instead.

[
  "run-prediction",
  "run-assessment",
  "assess-risk",
  "run-risk-assessment"
].forEach(id => {
  const button =
    document.getElementById(id);

  if (button) {
    button.addEventListener(
      "click",
      runPrediction
    );
  }
});

// X-ray button compatibility

[
  "analyze-xray",
  "xray-analyze",
  "run-xray"
].forEach(id => {
  const button =
    document.getElementById(id);

  if (button) {
    button.addEventListener(
      "click",
      analyzeXray
    );
  }
});

// ─────────────────────────────────────────────
// Backend Health Check
// ─────────────────────────────────────────────

async function checkBackendHealth() {
  try {
    const res =
      await fetch(
        "/health"
      );

    if (!res.ok) {
      throw new Error(
        "Backend unhealthy"
      );
    }

    const data =
      await res.json();

    console.log(
      "MedAI backend:",
      data
    );

  } catch (err) {
    console.warn(
      "Backend health check failed:",
      err
    );
  }
}

checkBackendHealth();

console.log(
  "MedAI frontend initialized."
);