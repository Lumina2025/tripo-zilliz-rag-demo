const form = document.querySelector("#searchForm");
const grid = document.querySelector("#resultsGrid");
const template = document.querySelector("#cardTemplate");
const statusText = document.querySelector("#statusText");
const statusDot = document.querySelector("#statusDot");
const submitBtn = document.querySelector("#submitBtn");
const fileInput = form.querySelector('input[type="file"]');
const fileName = document.querySelector("#fileName");

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files[0]?.name || "Drop or choose an image";
});

function setStatus(label, state = "ready") {
  statusText.textContent = label;
  statusDot.className = state === "ready" ? "" : state;
}

function chip(text) {
  const span = document.createElement("span");
  span.className = "chip";
  span.textContent = text;
  return span;
}

function renderResults(results) {
  grid.innerHTML = "";
  if (!results.length) {
    grid.innerHTML = '<div class="empty">No matching assets found.</div>';
    return;
  }

  for (const item of results) {
    const card = template.content.firstElementChild.cloneNode(true);
    const image = card.querySelector(".thumb");
    const link = card.querySelector(".thumb-link");
    const rank = card.querySelector(".rank");
    const score = card.querySelector(".score");
    const title = card.querySelector("h3");
    const caption = card.querySelector("p");
    const chips = card.querySelector(".chips");
    const paths = card.querySelector(".paths");

    image.src = item.render_url;
    image.alt = item.caption || item.llm_object || "Tripo render image";
    link.href = item.url || item.render_url;
    rank.textContent = `#${item.rank}`;
    score.textContent = `Score ${Number(item.score).toFixed(4)}`;
    title.textContent = item.llm_object || item.llm_keyword || item.project_id;
    caption.textContent = item.caption || "No caption available.";

    [item.llm_category, item.llm_style, item.llm_use_case, item.generation_mode]
      .filter(Boolean)
      .forEach((value) => chips.appendChild(chip(value)));

    paths.textContent = `milvus_render_images/${item.render_image_file}`;
    grid.appendChild(card);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  submitBtn.disabled = true;
  setStatus("Searching vector space", "busy");

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Search failed");
    }
    renderResults(payload.results);
    if (payload.raw_count && payload.raw_count !== payload.results.length) {
      setStatus(`Found ${payload.results.length} assets after score filtering`, "ready");
    } else {
      setStatus(`Found ${payload.results.length} assets`, "ready");
    }
    document.querySelector("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setStatus("Search failed", "error");
    console.error(error);
  } finally {
    submitBtn.disabled = false;
  }
});

form.querySelector('textarea[name="text"]').value = "fantasy sword with blue gemstone";
