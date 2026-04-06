"""Dashboard theme: custom CSS, colors, and visual constants."""

# Status colors
COLOR_SUCCESS = "#10b981"
COLOR_WARNING = "#f59e0b"
COLOR_ERROR = "#ef4444"
COLOR_INFO = "#3b82f6"
COLOR_MUTED = "#6b7280"

# Custom CSS injected into gr.Blocks
CUSTOM_CSS = """
.status-running {
    color: #3b82f6;
    font-weight: 600;
}
.status-complete {
    color: #10b981;
    font-weight: 600;
}
.status-failed {
    color: #ef4444;
    font-weight: 600;
}
.status-idle {
    color: #6b7280;
}
.metric-card {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-size: 2em;
    font-weight: 700;
    color: #1e293b;
}
.metric-label {
    font-size: 0.85em;
    color: #64748b;
    margin-top: 4px;
}
.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: #94a3b8;
}
.empty-state h3 {
    color: #64748b;
    margin-bottom: 8px;
}
.tab-header {
    border-bottom: 2px solid #f1f5f9;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
footer {visibility: hidden}
"""

EMPTY_STATE_TEMPLATE = """
<div class="empty-state">
<h3>{icon} {title}</h3>
<p>{message}</p>
{action}
</div>
"""


def empty_state(title: str, message: str, command: str = "", icon: str = "") -> str:
    """Generate a styled empty state message."""
    action = f"<p><code>{command}</code></p>" if command else ""
    return EMPTY_STATE_TEMPLATE.format(icon=icon, title=title, message=message, action=action)


def metric_card(value: str, label: str) -> str:
    """Generate HTML for a metric display card."""
    return f"""
<div class="metric-card">
    <div class="metric-value">{value}</div>
    <div class="metric-label">{label}</div>
</div>
"""


def status_badge(text: str, status: str = "idle") -> str:
    """Generate a color-coded status badge."""
    return f'<span class="status-{status}">{text}</span>'
