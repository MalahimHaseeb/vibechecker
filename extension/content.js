chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "get_comments") {
    const nodes = document.querySelectorAll("ytd-comment-thread-renderer #content-text");
    const comments = Array.from(nodes)
      .map((n) => n.innerText.trim())
      .filter(Boolean);
    sendResponse({ comments });
  }
});