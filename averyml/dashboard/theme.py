"""Dashboard theme: custom CSS, colors, HTML components."""

COLOR_SUCCESS = "#10b981"
COLOR_WARNING = "#f59e0b"
COLOR_ERROR = "#ef4444"
COLOR_INFO = "#3b82f6"
COLOR_MUTED = "#6b7280"
COLOR_ACCENT = "#f97316"

CUSTOM_CSS = """
/* ---- Global ---- */
footer {visibility: hidden}
.gradio-container { max-width: 1400px !important; }

/* ---- Hero ---- */
.hero {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 60%, #1e1b4b 100%);
    border-radius: 16px;
    padding: 32px 40px;
    color: white;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(249,115,22,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { font-size: 2em; margin: 0 0 4px 0; position: relative; }
.hero .ver { font-size: 0.4em; color: #94a3b8; font-weight: 400; vertical-align: middle; margin-left: 8px; }
.hero .tag { color: #cbd5e1; font-size: 1.05em; margin: 0; position: relative; }
.hero .acc { color: #f97316; font-weight: 600; }

/* ---- Metric Cards ---- */
.mc { background: linear-gradient(135deg, #fff 0%, #f8fafc 100%); border: 1px solid #e2e8f0;
      border-radius: 14px; padding: 20px 24px; text-align: center;
      transition: transform 0.15s ease, box-shadow 0.15s ease; }
.mc:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.06); }
.mc-icon { font-size: 1.4em; margin-bottom: 4px; }
.mc-val { font-size: 2.2em; font-weight: 800; color: #0f172a; line-height: 1.1; }
.mc-lbl { font-size: 0.8em; color: #64748b; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.5px; }

/* ---- Status Dots ---- */
.sd { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.sd.running { background: #3b82f6; animation: pulse 1.5s ease-in-out infinite; }
.sd.complete { background: #10b981; }
.sd.failed { background: #ef4444; }
.sd.idle { background: #cbd5e1; }
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.5; transform:scale(1.4); } }

.status-running { color: #3b82f6; font-weight: 600; }
.status-complete { color: #10b981; font-weight: 600; }
.status-failed { color: #ef4444; font-weight: 600; }
.status-idle { color: #6b7280; }

/* ---- Empty States ---- */
.es { text-align: center; padding: 48px 24px; background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
      border-radius: 16px; border: 2px dashed #e2e8f0; margin: 12px 0; }
.es .es-icon { font-size: 2.5em; margin-bottom: 10px; }
.es h3 { color: #475569; margin: 0 0 8px 0; font-size: 1.15em; }
.es p { color: #94a3b8; margin: 4px 0; }
.es code { display: inline-block; background: #1e293b; color: #f97316;
           padding: 8px 16px; border-radius: 8px; font-size: 0.88em; margin-top: 12px; }

/* ---- Highlight Card (best result) ---- */
.hl { background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); border: 1px solid #bbf7d0;
      border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; }
.hl .hl-lbl { font-size: 0.78em; color: #16a34a; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
.hl .hl-val { font-size: 1.6em; font-weight: 800; color: #14532d; }
.hl .hl-det { font-size: 0.85em; color: #4ade80; }

/* ---- Pipeline Steps ---- */
.steps { display: flex; gap: 10px; margin: 16px 0; }
.step { flex: 1; text-align: center; padding: 14px 10px; background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; transition: all 0.15s ease; }
.step:hover { border-color: #f97316; background: #fff7ed; }
.step-num { font-size: 1.6em; font-weight: 800; color: #f97316; }
.step-lbl { font-size: 0.78em; color: #64748b; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.3px; }
"""


def hero_banner() -> str:
    import averyml
    return f"""
<div class="hero">
    <h1>AveryML<span class="ver">v{averyml.__version__}</span></h1>
    <p class="tag">Make your LLM better at code by feeding it its own homework &mdash;
    <span class="acc">no answers required</span></p>
</div>"""


def metric_card(value: str, label: str, icon: str = "") -> str:
    icon_html = f'<div class="mc-icon">{icon}</div>' if icon else ""
    return f'<div class="mc">{icon_html}<div class="mc-val">{value}</div><div class="mc-lbl">{label}</div></div>'


def empty_state(title: str, message: str, command: str = "", icon: str = "") -> str:
    icon_html = f'<div class="es-icon">{icon}</div>' if icon else ""
    cmd_html = f"<code>{command}</code>" if command else ""
    return f'<div class="es">{icon_html}<h3>{title}</h3><p>{message}</p>{cmd_html}</div>'


def status_badge(text: str, status: str = "idle") -> str:
    return f'<span class="sd {status}"></span><span class="status-{status}">{text}</span>'


def highlight_card(label: str, value: str, detail: str = "") -> str:
    det = f'<div class="hl-det">{detail}</div>' if detail else ""
    return f'<div class="hl"><div class="hl-lbl">{label}</div><div class="hl-val">{value}</div>{det}</div>'


def pipeline_steps() -> str:
    return """
<div class="steps">
    <div class="step"><div class="step-num">1</div><div class="step-lbl">Synthesize</div></div>
    <div class="step"><div class="step-num">2</div><div class="step-lbl">Train</div></div>
    <div class="step"><div class="step-num">3</div><div class="step-lbl">Evaluate</div></div>
    <div class="step"><div class="step-num">4</div><div class="step-lbl">Analyze</div></div>
</div>"""
