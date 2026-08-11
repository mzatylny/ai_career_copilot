const state = { sessionId: null };

function headers(json = true) {
  const output = json ? { "Content-Type": "application/json" } : {};
  const key = document.querySelector("#api-key").value.trim();
  if (key) output["X-API-Key"] = key;
  return output;
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function render(target, value) {
  document.querySelector(target).textContent =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

document.querySelector("#create-session").addEventListener("click", async (event) => {
  event.currentTarget.disabled = true;
  try {
    const result = await request("/api/sessions", { method: "POST", headers: headers() });
    state.sessionId = result.session_id;
    render("#session-output", `Session ${state.sessionId}`);
    render("#upload-result", "Ready for a PDF.");
  } catch (error) {
    render("#session-output", error.message);
  } finally {
    event.currentTarget.disabled = false;
  }
});

document.querySelector("#gap-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  render("#gap-result", "Analyzing evidence...");
  try {
    const result = await request("/api/analyze-gap", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({
        resume_text: document.querySelector("#resume-text").value,
        job_description_text: document.querySelector("#job-text").value,
        target_seniority: "senior",
      }),
    });
    render("#gap-result", result);
  } catch (error) {
    render("#gap-result", error.message);
  } finally {
    button.disabled = false;
  }
});

async function pollJob(jobId) {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    const job = await request(`/api/jobs/${jobId}`, { headers: headers(false) });
    render("#upload-result", job);
    if (["completed", "failed"].includes(job.status)) return;
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
  throw new Error("Indexing is still running. Check the job endpoint later.");
}

document.querySelector("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.sessionId) return render("#upload-result", "Create a session first.");
  const button = event.submitter;
  button.disabled = true;
  const form = new FormData();
  form.append("session_id", state.sessionId);
  form.append("file", document.querySelector("#pdf-file").files[0]);
  try {
    const job = await request("/api/upload-document-async", {
      method: "POST",
      headers: headers(false),
      body: form,
    });
    render("#upload-result", job);
    await pollJob(job.job_id);
  } catch (error) {
    render("#upload-result", error.message);
  } finally {
    button.disabled = false;
  }
});

document.querySelector("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.sessionId) return render("#chat-result", "Create a session first.");
  const button = event.submitter;
  button.disabled = true;
  render("#chat-result", "Searching trusted context...");
  try {
    const result = await request("/api/chat", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({
        session_id: state.sessionId,
        user_query: document.querySelector("#chat-query").value,
      }),
    });
    render("#chat-result", result);
  } catch (error) {
    render("#chat-result", error.message);
  } finally {
    button.disabled = false;
  }
});
