from config import CONFIG, FALL_ASPECT


def classify_behavior(
    speed,
    overlaps,
    close_count,
    bbox,
    fight_hold,
    run_hold,
    fall_hold,
    pose_ok,
):
    """
    Rule-based behavior classifier.
    """

    x1, y1, x2, y2 = bbox

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    aspect = w / float(h)

    # Ignore impossible one-frame jumps.
    if speed > 240:
        return "UNKNOWN"

    # FALL
    if fall_hold >= CONFIG["fall_hold"] and speed <= 180:
        return "FALL"

    if pose_ok and aspect >= FALL_ASPECT and speed <= 180:
        return "FALL"

    # FIGHT
    if (
        overlaps >= 1
        and close_count >= 1
        and fight_hold >= max(CONFIG["fight_hold"], 5)
        and 45 <= speed <= 180
    ):
        return "FIGHT"

    # RUN
    run_speed = max(CONFIG["speed_th"], 120)

    if (
        overlaps == 0
        and close_count == 0
        and run_hold >= max(CONFIG["run_hold"], 3)
        and run_speed <= speed <= 220
    ):
        return "RUN"

    return "UNKNOWN"