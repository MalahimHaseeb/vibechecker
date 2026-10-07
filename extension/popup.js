const btn = document.getElementById("analyze");
const out = document.getElementById("result");

btn.addEventListener("click", async () => {
  out.textContent = "Reading comments...";
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.tabs.sendMessage(tab.id, { type: "get_comments" }, (res) => {
    if (chrome.runtime.lastError || !res || res.comments.length === 0) {
      out.textContent = "No comments found. Open a video and scroll down to the comments first.";
      return;
    }

    out.textContent = "Analyzing " + res.comments.length + " comments...";

    chrome.runtime.sendMessage({ type: "analyze", comments: res.comments }, (reply) => {
      if (!reply || !reply.ok) {
        out.textContent = "API error. Is the server running?";
        return;
      }
      const c = reply.data.counts;
      const t = reply.data.total;
      const pct = (n) => Math.round((n / t) * 100);
      out.innerHTML =
        "Positive: " + pct(c.positive) + "%<br>" +
        "Neutral: " + pct(c.neutral) + "%<br>" +
        "Negative: " + pct(c.negative) + "%<br>" +
        "Based on " + t + " comments";
    });
  });
});