import fs from "fs";
import path from "path";
import sharp from "file:///C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp/lib/index.js";

const root = "D:/code/Postgraduate-dissertation";
const report = path.join(root, "results/reports/thesis_final_evidence_figures_v1");
const outDir = path.join(root, "thesis/assets");
fs.mkdirSync(outDir, { recursive: true });

const labels = {
  vit_baseline: "No PE",
  vit_learnable_position: "Learnable PE",
  vit_multiplicative_sinusoidal_shifted: "Shifted Multiplicative PE",
  vit_row_col_mean_fusion: "Mean Fusion",
  vit_row_col_mean_mlp_fusion: "Mean + MLP Fusion",
  vit_row_col_latent_fusion: "Concat + MLP Fusion",
  vit_row_col_cross_attention_fusion: "Bidirectional Cross-Attention",
  vit_row_col_cross_attention_mlp_head_fusion: "Cross-Attention + MLP Head",
};

const colors = {
  vit_baseline: "#4b5563",
  vit_learnable_position: "#0072B2",
  vit_multiplicative_sinusoidal_shifted: "#E66101",
  vit_row_col_mean_fusion: "#117733",
  vit_row_col_mean_mlp_fusion: "#B8860B",
  vit_row_col_latent_fusion: "#7A3DB8",
  vit_row_col_cross_attention_fusion: "#332288",
  vit_row_col_cross_attention_mlp_head_fusion: "#CC6677",
};

const dashes = {
  vit_baseline: "",
  vit_learnable_position: "",
  vit_multiplicative_sinusoidal_shifted: "24 12",
  vit_row_col_mean_fusion: "24 12",
  vit_row_col_mean_mlp_fusion: "30 10 5 10",
  vit_row_col_latent_fusion: "5 10",
  vit_row_col_cross_attention_fusion: "24 12",
  vit_row_col_cross_attention_mlp_head_fusion: "5 10",
};

function csvRows(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const head = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(head.map((key, index) => [key, values[index]]));
  });
}

function esc(text) {
  return String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function group(rows) {
  const result = new Map();
  for (const row of rows) {
    const key = `${row.condition}|||${row.series}`;
    if (!result.has(key)) result.set(key, []);
    result.get(key).push({
      epoch: Number(row.epoch),
      mean: Number(row.mean),
      low: Number(row.ci95_lower),
      high: Number(row.ci95_upper),
    });
  }
  for (const values of result.values()) values.sort((a, b) => a.epoch - b.epoch);
  return result;
}

function pathPoints(points, x, y, field) {
  return points.map((p, i) => `${i ? "L" : "M"}${x(p.epoch).toFixed(1)},${y(p[field]).toFixed(1)}`).join(" ");
}

function panel({ x0, y0, width, height, title, series, yMin, yMax, showLegend = false }) {
  const margin = { left: 105, right: 28, top: 82, bottom: 90 };
  const px0 = x0 + margin.left;
  const py0 = y0 + margin.top;
  const pw = width - margin.left - margin.right;
  const ph = height - margin.top - margin.bottom;
  const all = series.flatMap((s) => s.rows);
  const maxEpoch = Math.max(...all.map((p) => p.epoch));
  const x = (epoch) => px0 + ((epoch - 1) / Math.max(1, maxEpoch - 1)) * pw;
  const y = (value) => py0 + ph - ((value - yMin) / (yMax - yMin)) * ph;
  let svg = `<text x="${x0 + width / 2}" y="${y0 + 38}" text-anchor="middle" class="panel-title">${esc(title)}</text>`;
  for (let tick = Math.ceil(yMin / 10) * 10; tick <= yMax; tick += 10) {
    const ty = y(tick);
    svg += `<line x1="${px0}" y1="${ty}" x2="${px0 + pw}" y2="${ty}" class="grid"/>`;
    svg += `<text x="${px0 - 20}" y="${ty + 10}" text-anchor="end" class="tick">${tick}</text>`;
  }
  svg += `<line x1="${px0}" y1="${py0}" x2="${px0}" y2="${py0 + ph}" class="axis"/>`;
  svg += `<line x1="${px0}" y1="${py0 + ph}" x2="${px0 + pw}" y2="${py0 + ph}" class="axis"/>`;
  for (let tick = 0; tick <= maxEpoch; tick += 10) {
    const tx = px0 + (tick / maxEpoch) * pw;
    svg += `<text x="${tx}" y="${py0 + ph + 38}" text-anchor="middle" class="tick">${tick}</text>`;
  }
  svg += `<text x="${px0 + pw / 2}" y="${py0 + ph + 78}" text-anchor="middle" class="label">Epoch</text>`;
  svg += `<text x="${x0 + 30}" y="${py0 + ph / 2}" text-anchor="middle" transform="rotate(-90 ${x0 + 30} ${py0 + ph / 2})" class="label">Validation accuracy (%)</text>`;
  for (const item of series) {
    const upper = item.rows.map((p) => `${x(p.epoch).toFixed(1)},${y(p.high).toFixed(1)}`).join(" ");
    const lower = [...item.rows].reverse().map((p) => `${x(p.epoch).toFixed(1)},${y(p.low).toFixed(1)}`).join(" ");
    svg += `<polygon points="${upper} ${lower}" fill="${colors[item.model]}" opacity="0.16"/>`;
    svg += `<path d="${pathPoints(item.rows, x, y, "mean")}" fill="none" stroke="${colors[item.model]}" stroke-width="6" stroke-dasharray="${dashes[item.model]}" stroke-linecap="round" stroke-linejoin="round"/>`;
  }
  if (showLegend) {
    const lx = px0 + pw - 470;
    let ly = py0 + ph - series.length * 42 - 15;
    svg += `<rect x="${lx - 25}" y="${ly - 30}" width="495" height="${series.length * 42 + 35}" fill="white" opacity="0.88"/>`;
    for (const item of series) {
      svg += `<line x1="${lx}" y1="${ly}" x2="${lx + 72}" y2="${ly}" stroke="${colors[item.model]}" stroke-width="6" stroke-dasharray="${dashes[item.model]}"/>`;
      svg += `<text x="${lx + 92}" y="${ly + 10}" class="legend">${esc(labels[item.model])}</text>`;
      ly += 42;
    }
  }
  return svg;
}

function baseSvg(width, height, title, content) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="100%" height="100%" fill="white"/>
  <style>
    text { font-family: Arial, sans-serif; fill: #1f2937; }
    .title { font-size: 54px; font-weight: 400; }
    .panel-title { font-size: 42px; }
    .label { font-size: 32px; }
    .tick { font-size: 28px; }
    .legend { font-size: 28px; }
    .grid { stroke: #e5e7eb; stroke-width: 3; stroke-dasharray: 10 8; }
    .axis { stroke: #cbd5e1; stroke-width: 4; }
  </style>
  <text x="${width / 2}" y="65" text-anchor="middle" class="title">${esc(title)}</text>
  ${content}
  </svg>`;
}

async function writeGraphic(name, svg) {
  const svgPath = path.join(outDir, `${name}.svg`);
  const pngPath = path.join(outDir, `${name}.png`);
  fs.writeFileSync(svgPath, svg, "utf8");
  await sharp(Buffer.from(svg)).png().toFile(pngPath);
  console.log(svgPath);
  console.log(pngPath);
}

async function lowData() {
  const curves = group(csvRows(path.join(report, "low_data_validation_epoch_summary.csv")));
  const width = 3600, height = 2400;
  const models = ["vit_baseline", "vit_learnable_position", "vit_multiplicative_sinusoidal_shifted"];
  const specs = [["1000", "1k training examples"], ["5000", "5k training examples"], ["10000", "10k training examples"], ["45000", "Full (45k) training examples"]];
  const all = specs.flatMap(([condition]) => models.flatMap((model) => curves.get(`${condition}|||${model}`)));
  const yMin = Math.floor((Math.min(...all.map((p) => p.low)) - 4) / 10) * 10;
  const yMax = Math.ceil((Math.max(...all.map((p) => p.high)) + 4) / 10) * 10;
  let content = "";
  specs.forEach(([condition, title], index) => {
    const x0 = index % 2 === 0 ? 40 : 1810;
    const y0 = index < 2 ? 110 : 1210;
    content += panel({ x0, y0, width: 1750, height: 1050, title, yMin, yMax, series: models.map((model) => ({ model, rows: curves.get(`${condition}|||${model}`) })) });
  });
  let lx = 800;
  for (const model of models) {
    content += `<line x1="${lx}" y1="2350" x2="${lx + 80}" y2="2350" stroke="${colors[model]}" stroke-width="6" stroke-dasharray="${dashes[model]}"/>`;
    content += `<text x="${lx + 100}" y="2360" class="legend">${esc(labels[model])}</text>`;
    lx += model === "vit_multiplicative_sinusoidal_shifted" ? 0 : 780;
  }
  await writeGraphic("low_data_validation_accuracy_epoch_valid", baseSvg(width, height, "CIFAR-10 training-set size: validation trajectories", content));
}

async function fusion() {
  const curves = group(csvRows(path.join(report, "fusion_validation_epoch_summary.csv")));
  const width = 3600, height = 1500;
  const left = ["vit_learnable_position", "vit_row_col_mean_fusion", "vit_row_col_mean_mlp_fusion", "vit_row_col_latent_fusion"];
  const right = ["vit_learnable_position", "vit_row_col_cross_attention_fusion", "vit_row_col_cross_attention_mlp_head_fusion"];
  const all = [...left, ...right].flatMap((model) => curves.get(`fusion|||${model}`));
  const yMin = Math.floor((Math.min(...all.map((p) => p.low)) - 3) / 5) * 5;
  const yMax = Math.ceil((Math.max(...all.map((p) => p.high)) + 3) / 5) * 5;
  let content = panel({ x0: 35, y0: 110, width: 1740, height: 1330, title: "(a) Aggregation-based fusion", yMin, yMax, showLegend: true, series: left.map((model) => ({ model, rows: curves.get(`fusion|||${model}`) })) });
  content += panel({ x0: 1800, y0: 110, width: 1765, height: 1330, title: "(b) Cross-attention fusion", yMin, yMax, showLegend: true, series: right.map((model) => ({ model, rows: curves.get(`fusion|||${model}`) })) });
  await writeGraphic("fusion_validation_accuracy_epoch_valid", baseSvg(width, height, "Single-branch reference and dual-branch fusion: validation trajectories", content));
}

await lowData();
await fusion();
