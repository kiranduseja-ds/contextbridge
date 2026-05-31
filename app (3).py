import gradio as gr
from groq import Groq
import json
import os

SYSTEM_PROMPT = """You are ContextBridge, an expert communication analyst. Analyze workplace communication and return ONLY a JSON object — no extra text, no markdown, no backticks, just raw JSON.

Return exactly this structure:
{
  "risk_score": <integer 0-100>,
  "risk_label": "<Low|Medium|High|Critical>",
  "what_writer_meant": "<1-2 sentences explaining the real intent>",
  "what_reader_will_think": [
    "<misinterpretation 1>",
    "<misinterpretation 2>",
    "<misinterpretation 3>"
  ],
  "problem_phrases": [
    "<exact phrase from text causing confusion>",
    "<exact phrase from text causing confusion>"
  ],
  "rewrite": "<full rewritten version that preserves the writer's tone>",
  "explanation": "<1 sentence on why this rewrite is clearer>"
}

Risk score guide:
0-30 = Low: clear communication
31-60 = Medium: some ambiguity, could cause delays
61-85 = High: likely to be misunderstood
86-100 = Critical: will almost certainly cause miscommunication

Be specific and practical. Rewrite must sound like the original writer — not formal, not robotic. Return ONLY the JSON, nothing else."""


def analyze(text, api_key):
    if not text.strip():
        return "Please enter some text to analyze.", "", "", "", ""

    key = api_key.strip() if api_key.strip() else os.environ.get("GROQ_API_KEY", "")
    if not key:
        return "Please enter your Groq API key. Get one free at console.groq.com", "", "", "", ""

    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this workplace communication:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        raw = response.choices[0].message.content.strip()

        # Remove markdown code fences if model adds them
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        raw = raw.strip()
        data = json.loads(raw)

        score = data["risk_score"]
        label = data["risk_label"]

        if score <= 30:
            color = "🟢"
        elif score <= 60:
            color = "🟡"
        elif score <= 85:
            color = "🔴"
        else:
            color = "🚨"

        score_out = f"{color} {score}/100 — {label} Miscommunication Risk"
        intent_out = f"What the writer actually meant:\n{data['what_writer_meant']}"

        misread_out = "How readers will misinterpret this:\n"
        for i, m in enumerate(data["what_reader_will_think"], 1):
            misread_out += f"\n{i}. {m}"

        phrases_out = "Problem phrases causing confusion:\n"
        for p in data["problem_phrases"]:
            phrases_out += f'\n→ "{p}"'

        rewrite_out = f"Rewritten version:\n\n{data['rewrite']}\n\n---\nWhy this works: {data['explanation']}"

        return score_out, intent_out, misread_out, phrases_out, rewrite_out

    except json.JSONDecodeError as e:
        return f"Parsing error: {str(e)} — Try again.", "", "", "", ""
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", ""


EXAMPLES = [
    ["Fix the thing from last week asap its blocking everything"],
    ["Can you look at the doc I sent and let me know"],
    ["The client is not happy. Please handle it."],
    ["We need to discuss the project. It's urgent."],
    ["The feature needs to be done differently than we discussed"],
]

with gr.Blocks(
    title="ContextBridge — AI Communication Gap Detector",
    theme=gr.themes.Soft(),
    css="""
    .risk-box textarea { font-size: 18px !important; font-weight: 600 !important; }
    .rewrite-box textarea { background: #f0fdf4 !important; }
    footer { display: none !important; }
    """
) as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 24px 0 8px;">
        <h1 style="font-size:28px; font-weight:700; margin-bottom:8px;">ContextBridge</h1>
        <p style="color:#6b7280; font-size:15px; max-width:520px; margin:0 auto;">
            The world's first AI that detects the gap between what you <em>meant</em> and what you <em>wrote</em> —
            then predicts exactly how your reader will misunderstand it.
        </p>
    </div>
    """)

    api_input = gr.Textbox(
        label="Groq API Key — free at console.groq.com (no credit card needed)",
        placeholder="gsk_...",
        type="password",
        lines=1
    )
    text_input = gr.Textbox(
        label="Paste any workplace communication — email, Jira ticket, Slack message, requirement",
        placeholder="e.g. Fix the thing from last week asap its blocking prod",
        lines=6
    )
    analyze_btn = gr.Button("Detect Communication Gap", variant="primary", size="lg")

    gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try these real examples")

    gr.HTML("<hr style='margin:16px 0; border-color:#e5e7eb;'>")

    with gr.Row():
        with gr.Column():
            score_out = gr.Textbox(label="Miscommunication Risk Score", elem_classes=["risk-box"], lines=1)
            intent_out = gr.Textbox(label="What the writer actually meant", lines=3)
        with gr.Column():
            misread_out = gr.Textbox(label="How readers will misinterpret this", lines=5)
            phrases_out = gr.Textbox(label="Problem phrases identified", lines=3)

    rewrite_out = gr.Textbox(
        label="ContextBridge Rewrite — in the writer's own voice",
        lines=5,
        elem_classes=["rewrite-box"]
    )

    gr.HTML("""
    <div style="text-align:center; padding:16px; color:#9ca3af; font-size:12px;">
        Built by Kiran Duseja · NLP Portfolio Project ·
        <a href="https://github.com/yourusername/contextbridge" target="_blank">GitHub</a>
    </div>
    """)

    analyze_btn.click(
        fn=analyze,
        inputs=[text_input, api_input],
        outputs=[score_out, intent_out, misread_out, phrases_out, rewrite_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)
