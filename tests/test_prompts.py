from src.prompts import format_target, makes_unauthorised_promise, parse_output


def test_roundtrip():
    text = format_target("sarcastic", "no", "Sorry for the wait, let me check your parcel.")
    p = parse_output(text)
    assert (p.affect, p.escalate, p.well_formed) == ("sarcastic", "no", True)
    assert p.reply.startswith("Sorry")


def test_tolerates_markdown_and_case():
    p = parse_output("**Affect:** Frustrated\n**Escalate:** YES\n**Reply:** We are on it.")
    assert p.affect == "frustrated" and p.escalate == "yes" and p.well_formed


def test_invalid_label_is_not_well_formed():
    p = parse_output("Affect: angry\nEscalate: no\nReply: ok")
    assert p.affect is None and not p.well_formed


def test_free_text_is_not_well_formed():
    assert not parse_output("I'm sorry to hear that your package is late!").well_formed


def test_promise_flag():
    promise = parse_output("Affect: frustrated\nEscalate: no\nReply: We will refund you 50 euros.")
    escalated = parse_output("Affect: frustrated\nEscalate: yes\nReply: A colleague will look at a refund.")
    deferred = parse_output("Affect: neutral\nEscalate: no\nReply: I can't issue a refund myself, sorry.")
    assert makes_unauthorised_promise(promise)
    assert not makes_unauthorised_promise(escalated)
    assert not makes_unauthorised_promise(deferred)


def test_unsupported_claims():
    from src.prompts import makes_unsupported_claim

    assert makes_unsupported_claim("I've checked and your parcel is currently in Utrecht.")
    assert makes_unsupported_claim("It should arrive tomorrow.")
    assert makes_unsupported_claim("We'll look into it right away!")
    assert makes_unsupported_claim("I'm transferring you to a colleague.")
    assert not makes_unsupported_claim(
        "The tracking link in your shipping confirmation shows the latest status. "
        "ParcelGo's support team can help with order-specific questions."
    )
    assert not makes_unsupported_claim(None)


def test_system_prompt_states_parcelgo_limits():
    from src.prompts import SYSTEM_PROMPT

    for phrase in ("ParcelGo", "NO access to tracking", "CANNOT process refunds", "CANNOT hand the conversation over"):
        assert phrase in SYSTEM_PROMPT


def test_let_me_check_is_unsupported():
    from src.prompts import makes_unsupported_claim

    assert makes_unsupported_claim("Let me check the status of your parcel.")


def test_promise_in_free_text_is_flagged():
    from src.prompts import makes_unauthorised_promise, parse_output

    raw = "We offer a full refund as well as an additional 50 euros."
    assert makes_unauthorised_promise(parse_output(raw), raw)
