(() => {
  "use strict";

  const canvas = document.getElementById("world");
  const context = canvas.getContext("2d", { alpha: false });
  const stateNode = document.getElementById("state");
  const stageMessage = document.getElementById("stage-message");
  const fields = {
    action: document.getElementById("action"),
    actionRgb: document.getElementById("action-rgb"),
    browserE2E: document.getElementById("browser-e2e"),
    nextReady: document.getElementById("next-ready"),
    frame: document.getElementById("frame"),
    dropped: document.getElementById("dropped"),
    details: document.getElementById("details"),
  };

  const actions = new Map();
  const keyNames = new Map([
    ["w", "w"], ["s", "s"], ["a", "a"], ["d", "d"],
    ["j", "j"], ["l", "l"], ["i", "i"], ["k", "k"],
    [" ", " "], ["space", " "], ["spacebar", " "],
    ["escape", "escape"], ["esc", "escape"], ["q", "q"],
  ]);
  let socket = null;
  let nextActionId = 1;
  let currentActionId = 0;
  let currentActionKey = "bootstrap";
  let tailQueue = [];
  let tailTimer = null;
  let nextTailDue = 0;
  let droppedTails = 0;
  let frameDecodeChain = Promise.resolve();

  function now() { return performance.now(); }
  function send(message) {
    if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
  }
  function ms(value) { return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(1)} ms`; }

  function updateHud(record) {
    if (!record) return;
    actions.set(record.action_id, record);
    fields.action.textContent = record.action || "—";
    const server = record.server || {};
    const derived = record.derived || {};
    fields.actionRgb.textContent = ms(derived.action_to_base_rgb_ms);
    fields.nextReady.textContent = ms(derived.action_to_next_ready_ms);
    const browser = record.browser || {};
    const keydown = browser.keydown_ms;
    const presented = browser.frame_presented_ms;
    fields.browserE2E.textContent = keydown != null && presented != null ? ms(presented - keydown) : "—";
    fields.details.textContent = JSON.stringify(record, null, 2);
  }

  function setState(message) {
    stateNode.textContent = message.state || "UNKNOWN";
    stateNode.className = "state";
    if (message.state === "INTERACTIVE_READY") stateNode.classList.add("state-ready");
    if (message.state === "ERROR") stateNode.classList.add("state-error");
    stageMessage.textContent = message.text || message.state || "Loading…";
    if (message.state === "INTERACTIVE_READY" || message.state === "ACTION_RUNNING") stageMessage.classList.add("hidden");
    else if (message.state !== "STOPPING") stageMessage.classList.remove("hidden");
  }

  function recordBrowser(actionId, field, value) {
    send({ type: "browser_telemetry", action_id: actionId, field, value_ms: value });
  }

  function drawBitmap(item) {
    const drawAt = now();
    context.drawImage(item.bitmap, 0, 0, canvas.width, canvas.height);
    requestAnimationFrame(() => {
      const painted = now();
      fields.frame.textContent = `${item.actionId} · ${item.frameIndex + 1}/${item.frameCount}`;
      if (item.frameIndex === 0) recordBrowser(item.actionId, "frame_presented_ms", painted);
      const record = actions.get(item.actionId);
      if (record) {
        record.browser = record.browser || {};
        if (item.frameIndex === 0) record.browser.frame_presented_ms = painted;
        updateHud(record);
      }
      item.bitmap.close();
      void drawAt;
    });
  }

  function scheduleTail() {
    if (tailTimer !== null || tailQueue.length === 0) return;
    const delay = Math.max(0, nextTailDue - now());
    tailTimer = setTimeout(() => {
      tailTimer = null;
      const item = tailQueue.shift();
      if (item && item.actionId === currentActionId) {
        drawBitmap(item);
        nextTailDue = Math.max(nextTailDue + 62.5, now());
      } else if (item) {
        item.bitmap.close();
      }
      scheduleTail();
    }, delay);
  }

  async function receiveFrame(packet) {
    const view = new DataView(packet.buffer, packet.byteOffset, packet.byteLength);
    if (packet.byteLength < 4) return;
    const headerLength = view.getUint32(0, false);
    const headerStart = 4;
    const headerEnd = headerStart + headerLength;
    if (headerEnd > packet.byteLength) return;
    const header = JSON.parse(new TextDecoder().decode(packet.slice(headerStart, headerEnd)));
    const actionId = Number(header.action_id);
    const frameIndex = Number(header.frame_index);
    const received = now();
    if (frameIndex === 0) recordBrowser(actionId, "frame_received_ms", received);
    const blob = new Blob([packet.slice(headerEnd)], { type: "image/jpeg" });
    const bitmap = await createImageBitmap(blob);
    const decoded = now();
    if (frameIndex === 0) recordBrowser(actionId, "frame_decoded_ms", decoded);
    if (frameIndex === 0) {
      if (actionId !== currentActionId) {
        for (const old of tailQueue) old.bitmap.close();
        droppedTails += tailQueue.length;
        tailQueue = [];
        if (tailTimer !== null) { clearTimeout(tailTimer); tailTimer = null; }
      }
      currentActionId = actionId;
      currentActionKey = actions.get(actionId)?.action || currentActionKey;
      nextTailDue = now() + 62.5;
      drawBitmap({ actionId, frameIndex, frameCount: Number(header.frame_count), bitmap });
      stageMessage.classList.add("hidden");
    } else if (actionId === currentActionId) {
      if (tailQueue.length >= 3) {
        tailQueue.shift().bitmap.close();
        droppedTails += 1;
      }
      tailQueue.push({ actionId, frameIndex, frameCount: Number(header.frame_count), bitmap });
      scheduleTail();
    } else {
      bitmap.close();
      droppedTails += 1;
    }
    fields.dropped.textContent = String(droppedTails);
  }

  function onKeydown(event) {
    const key = keyNames.get(String(event.key).toLowerCase());
    if (key == null) return;
    event.preventDefault();
    if (event.repeat && key !== "q" && key !== "escape") return;
    const actionId = nextActionId++;
    const keydown = now();
    if (key === "q" || key === "escape") {
      send({ type: "action", action_id: actionId, key, client_keydown_ms: keydown });
      return;
    }
    actions.set(actionId, {
      action_id: actionId, action: key,
      browser: { keydown_ms: keydown }, server: {}, status: "submitted",
    });
    send({ type: "action", action_id: actionId, key, client_keydown_ms: keydown });
    // Send the action before its auxiliary browser telemetry. The server
    // creates the authoritative ledger record from the action request; if
    // telemetry arrives first it must not turn the action into a duplicate.
    recordBrowser(actionId, "keydown_ms", keydown);
  }

  function connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws`);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => setState({ state: "SERVER_STARTING", text: "Connected; waiting for model…" });
    socket.onclose = () => setState({ state: "STOPPING", text: "Connection closed" });
    socket.onerror = () => setState({ state: "ERROR", text: "WebSocket error" });
    socket.onmessage = async (event) => {
      if (typeof event.data === "string") {
        const message = JSON.parse(event.data);
        if (message.type === "status") setState(message);
        else if (message.type === "action_record") updateHud(message.record);
        else if (message.type === "input_ack" && message.status === "replaced") {
          const old = actions.get(message.replaced_action_id);
          if (old) { old.status = "replaced"; old.replaced_by = message.action_id; updateHud(old); }
        }
      } else {
        // Keep JPEG decode/presentation ordering deterministic. WebSocket
        // delivery is ordered, but createImageBitmap is asynchronous.
        frameDecodeChain = frameDecodeChain
          .then(() => receiveFrame(new Uint8Array(event.data)))
          .catch((error) => console.error("frame decode failed", error));
      }
    };
  }

  window.addEventListener("keydown", onKeydown, { passive: false });
  connect();
})();
