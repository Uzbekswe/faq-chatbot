const elements = {
  composer: document.getElementById("composer"),
  input: document.getElementById("message-input"),
  send: document.getElementById("send-button"),
  stop: document.getElementById("stop-button"),
  newChat: document.getElementById("new-chat"),
  messages: document.getElementById("messages"),
  empty: document.getElementById("empty-state"),
  setup: document.getElementById("setup-panel"),
  setupMissing: document.getElementById("setup-missing"),
  serviceStatus: document.getElementById("service-status"),
  statusLabel: document.getElementById("status-label"),
  template: document.getElementById("message-template"),
};

const state = { ready: false, streaming: false, controller: null };

function setStatus(kind, label) {
  elements.serviceStatus.dataset.state = kind;
  elements.statusLabel.textContent = label;
}

function setComposerEnabled(enabled) {
  elements.input.disabled = !enabled;
  elements.send.disabled = !enabled || state.streaming;
  elements.newChat.disabled = !enabled;
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.disabled = !enabled;
  });
}

function setStreaming(streaming) {
  state.streaming = streaming;
  elements.messages.setAttribute("aria-busy", String(streaming));
  elements.send.hidden = streaming;
  elements.stop.hidden = !streaming;
  elements.input.disabled = streaming || !state.ready;
  elements.send.disabled = streaming || !state.ready;
}

function scrollToLatest() {
  elements.messages.scrollTo({ top: elements.messages.scrollHeight, behavior: "smooth" });
}

function addMessage(role, content, options = {}) {
  if (elements.empty.contains(document.activeElement)) document.activeElement.blur();
  elements.empty.hidden = true;
  const fragment = elements.template.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  article.dataset.role = role;
  article.dataset.state = options.state || "complete";
  article.querySelector(".message-label").textContent = role === "user" ? "You" : "Assistant";
  article.querySelector(".message-body").textContent = content;
  if (options.id) article.dataset.messageId = options.id;
  elements.messages.append(article);
  scrollToLatest();
  return article;
}

function renderSources(article, sources) {
  if (!sources?.length) return;
  const details = article.querySelector(".sources");
  details.hidden = false;
  details.querySelector(".source-count").textContent = `(${sources.length})`;
  const list = details.querySelector(".source-list");
  list.replaceChildren();
  for (const source of sources) {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = source.question || "FAQ source";
    const excerpt = document.createElement("span");
    excerpt.textContent = source.answer_excerpt || source.answer || "";
    item.append(title, excerpt);
    if (typeof source.score === "number") {
      const score = document.createElement("span");
      score.className = "source-score";
      score.textContent = ` · Match ${Math.round(source.score * 100)}%`;
      item.append(score);
    }
    list.append(item);
  }
}

function parseSseBlock(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  const text = data.join("\n");
  try {
    return { event, data: JSON.parse(text) };
  } catch {
    return { event, data: { text } };
  }
}

async function consumeSse(response, onEvent) {
  if (!response.body) throw new Error("Streaming is not supported by this browser.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replaceAll("\r\n", "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseSseBlock(block);
      if (parsed) onEvent(parsed);
    }
    if (done) break;
  }
  const parsed = parseSseBlock(buffer.trim());
  if (parsed) onEvent(parsed);
}

async function ensureSession() {
  const response = await fetch("/api/v1/sessions", { method: "POST", credentials: "same-origin" });
  if (!response.ok) throw new Error("Could not start a private session.");
}

async function loadHistory() {
  const response = await fetch("/api/v1/sessions/current/messages", { credentials: "same-origin" });
  if (response.status === 401 || response.status === 404) return false;
  if (!response.ok) throw new Error("Could not restore conversation history.");
  const payload = await response.json();
  const messages = Array.isArray(payload) ? payload : payload.messages || [];
  for (const message of messages) {
    const article = addMessage(message.role, message.content, { id: message.id });
    renderSources(article, message.sources || []);
  }
  return true;
}

async function sendMessage(message) {
  if (!message || state.streaming || !state.ready) return;
  addMessage("user", message);
  elements.input.value = "";
  elements.input.style.height = "auto";
  const assistant = addMessage("assistant", "", { state: "streaming" });
  let sources = [];
  state.controller = new AbortController();
  setStreaming(true);

  try {
    const response = await fetch("/api/v1/chat/stream", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ message }),
      signal: state.controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail?.message || payload.detail || "The assistant is unavailable.");
    }

    await consumeSse(response, ({ event, data }) => {
      if (event === "sources") {
        sources = data.sources || data;
        renderSources(assistant, sources);
      } else if (event === "delta") {
        assistant.querySelector(".message-body").textContent += data.text || data.delta || "";
      } else if (event === "completed") {
        assistant.dataset.state = "complete";
        if (data.message_id) assistant.dataset.messageId = data.message_id;
      } else if (event === "error") {
        throw new Error(data.message || "The response stream failed.");
      }
      scrollToLatest();
    });
    assistant.dataset.state = "complete";
    renderSources(assistant, sources);
  } catch (error) {
    if (error.name === "AbortError") {
      assistant.dataset.state = "complete";
      if (!assistant.querySelector(".message-body").textContent) assistant.remove();
    } else {
      assistant.dataset.state = "error";
      assistant.querySelector(".message-body").textContent = error.message;
    }
  } finally {
    state.controller = null;
    setStreaming(false);
    elements.input.focus();
  }
}

async function resetConversation() {
  if (state.streaming) state.controller?.abort();
  await fetch("/api/v1/sessions/current", { method: "DELETE", credentials: "same-origin" });
  elements.messages.querySelectorAll(".message").forEach((message) => message.remove());
  elements.empty.hidden = false;
  await ensureSession();
  elements.input.focus();
}

async function initialize() {
  try {
    const response = await fetch("/api/v1/config", { credentials: "same-origin" });
    if (!response.ok) throw new Error("Service configuration could not be checked.");
    const config = await response.json();
    state.ready = Boolean(config.ready);
    if (state.ready) {
      setStatus("ready", "Ready");
      elements.setup.hidden = true;
      const restored = await loadHistory();
      if (!restored) await ensureSession();
    } else {
      setStatus("unconfigured", "Setup needed");
      elements.setup.hidden = false;
      const missing = config.missing || config.missing_configuration || [];
      elements.setupMissing.textContent = missing.length ? `Missing: ${missing.join(", ")}` : "Provider configuration is incomplete.";
    }
    setComposerEnabled(state.ready);
  } catch (error) {
    setStatus("error", "Unavailable");
    state.ready = false;
    elements.setup.hidden = false;
    elements.setupMissing.textContent = error.message;
    setComposerEnabled(false);
  }
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(elements.input.value.trim());
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

elements.input.addEventListener("input", () => {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 144)}px`;
});

elements.stop.addEventListener("click", () => state.controller?.abort());
elements.newChat.addEventListener("click", () => resetConversation().catch((error) => addMessage("assistant", error.message, { state: "error" })));
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

initialize();
