"""Live demo of the support agent for the presentation.

Terminal chat (default) or a small web UI with --gradio. Each turn shows the
detected affect and escalation recommendation next to the reply. When the
agent recommends escalation, the demo prints the case summary a human agent
would receive. The ParcelGo prototype has no operational hand-over, so the
summary is only displayed, not sent to a support system.

Usage:
    python -m src.demo --adapter outputs/support-lora/final
    python -m src.demo --adapter outputs/support-lora/final --base   # compare with base model
    python -m src.demo --adapter outputs/support-lora/final --gradio
"""

from __future__ import annotations

import argparse

from src.generation import SupportAgent
from src.prompts import DEFAULT_MODEL, build_messages, parse_output


def handoff_summary(history: list[dict], affect: str | None) -> str:
    customer_turns = [m["content"] for m in history if m["role"] == "user"]
    return (
        "--- RECOMMENDED HAND-OFF (prototype: shown here, not sent) ---\n"
        f"Detected affect: {affect or 'unknown'}\n"
        f"Customer turns: {len(customer_turns)}\n"
        f"Last message: {customer_turns[-1] if customer_turns else ''}\n"
        "For the human agent: check order records, investigate, decide on compensation.\n"
        "-------------------------------------------------------------"
    )


class Session:
    def __init__(self, agent: SupportAgent, use_adapter: bool):
        self.agent, self.use_adapter = agent, use_adapter
        self.history: list[dict] = []

    def turn(self, message: str) -> tuple[str, str]:
        raw = self.agent.generate(build_messages(message, self.history), use_adapter=self.use_adapter)
        parsed = parse_output(raw)
        self.history += [{"role": "user", "content": message}, {"role": "assistant", "content": raw}]
        status = f"[affect: {parsed.affect or '?'} | escalate: {parsed.escalate or '?'}]"
        reply = parsed.reply or raw
        if parsed.escalate == "yes":
            reply += "\n\n" + handoff_summary(self.history, parsed.affect)
        return status, reply


def run_cli(session: Session) -> None:
    print("ParcelGo support chatbot. Type 'quit' to exit, 'reset' to start a new conversation.\n")
    while True:
        try:
            msg = input("Customer> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {"quit", "exit"}:
            break
        if msg.lower() == "reset":
            session.history.clear()
            print("(conversation reset)\n")
            continue
        if not msg:
            continue
        status, reply = session.turn(msg)
        print(f"{status}\nAgent> {reply}\n")


def run_gradio(session: Session) -> None:
    import gradio as gr

    def respond(message, chat):
        status, reply = session.turn(message)
        return chat + [{"role": "user", "content": message},
                       {"role": "assistant", "content": f"{status}\n\n{reply}"}], ""

    def reset():
        session.history.clear()
        return [], ""

    with gr.Blocks(title="ParcelGo support chatbot") as ui:
        gr.Markdown("### ParcelGo support chatbot\nDetected affect and escalation recommendation are shown above each reply.")
        chat = gr.Chatbot(height=480)  # gradio >= 6: messages format by default
        box = gr.Textbox(placeholder="Write as a customer with a delayed delivery…")
        box.submit(respond, [box, chat], [chat, box])
        gr.Button("New conversation").click(reset, None, [chat, box])
    ui.launch(share=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None)
    p.add_argument("--base", action="store_true", help="use the base model even if an adapter is loaded")
    p.add_argument("--gradio", action="store_true")
    args = p.parse_args()

    agent = SupportAgent(args.model, args.adapter)
    session = Session(agent, use_adapter=bool(args.adapter) and not args.base)
    run_gradio(session) if args.gradio else run_cli(session)


if __name__ == "__main__":
    main()
