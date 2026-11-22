import csv

from src.prepare_data import clean_text, heuristic_affect, heuristic_escalate, is_delay_complaint, load_pairs


def test_clean_text_removes_handles_urls_signatures():
    assert clean_text("@AmazonHelp where is it? https://t.co/x") == "where is it?"
    assert clean_text("@user123 Sorry about that! ^JD", strip_signature=True) == "Sorry about that!"


def test_delay_filter():
    assert is_delay_complaint("my package is 3 days late")
    assert not is_delay_complaint("how do I reset my password")


def test_heuristic_labels():
    assert heuristic_affect("Wow, a week late. Truly impressive 👏") == "sarcastic"
    assert heuristic_affect("This is ridiculous, my order is late") == "frustrated"
    assert heuristic_affect("Hi, when will my order arrive?") == "neutral"
    assert heuristic_escalate("I want a refund for this late order", "frustrated") == "yes"
    assert heuristic_escalate("When will my order arrive?", "neutral") == "no"


def test_load_pairs(tmp_path):
    path = tmp_path / "twcs.csv"
    rows = [
        # tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id
        [1, "115712", "True", "x", "@Shop my package is 4 days late, ridiculous", "2", ""],
        [2, "Shop", "False", "x", "@115712 Sorry for the delay! Please DM us your order number. ^AB", "", "1"],
        [3, "115713", "True", "x", "@Shop how do I change my password", "4", ""],
        [4, "Shop", "False", "x", "@115713 Go to settings and follow the steps there.", "", "3"],
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"])
        w.writerows(rows)
    pairs = load_pairs(path)
    assert len(pairs) == 1
    assert pairs.iloc[0]["customer_text"] == "my package is 4 days late, ridiculous"
    assert pairs.iloc[0]["reply_text"] == "Sorry for the delay! Please DM us your order number."


def test_load_pairs_drops_unsupported_claims(tmp_path):
    path = tmp_path / "twcs.csv"
    rows = [
        [1, "a", "True", "x", "@Shop my package is 4 days late", "2", ""],
        [2, "Shop", "False", "x", "@a I've checked and it will arrive tomorrow!", "", "1"],
        [3, "b", "True", "x", "@Shop my order is still not delivered", "4", ""],
        [4, "Shop", "False", "x", "@b Sorry about that! Please DM us so our team can help.", "", "3"],
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"])
        w.writerows(rows)
    assert len(load_pairs(path)) == 1
    assert len(load_pairs(path, keep_unsupported_claims=True)) == 2
