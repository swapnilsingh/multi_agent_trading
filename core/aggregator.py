from collections import Counter

def aggregate_actions(votes):
    """
    Aggregates actions from different agents.
    Priority: Majority > Any non-zero > Hold
    """
    vote_counter = Counter(votes)

    # Majority rule
    if vote_counter[1] > vote_counter[0] and vote_counter[1] > vote_counter[-1]:
        return 1  # BUY
    elif vote_counter[-1] > vote_counter[0] and vote_counter[-1] > vote_counter[1]:
        return -1  # SELL

    # If there's no majority but at least one vote
    if 1 in votes:
        return 1
    elif -1 in votes:
        return -1

    return 0  # HOLD
