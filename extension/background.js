const API_URL = "http://localhost:8000/predict";

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "analyze") {
    fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: msg.comments }),
    })
      .then((r) => r.json())
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }
});