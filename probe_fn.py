"""
Key probe (Windows).

Prints the raw name and scan code of every key you press, then tells you
exactly what to put in config.json to bind it. Use it whenever a key does
not trigger Nightjar - the name or scan code the driver actually emits is
often not the one you would guess (Right Alt is a classic example, and Fn
on most laptops emits nothing at all).

Run it, press the key you care about a few times, then press ESC three times.
"""

import keyboard

seen: dict[tuple[str, int], int] = {}
order: list[tuple[str, int]] = []
esc_presses = 0
finished = False

# Keys that are only ever pressed *with* something else - never worth
# suggesting as a standalone hotkey, and they crowd the summary.
IGNORE = {"esc"}

print("=" * 64)
print("  KEY PROBE")
print("=" * 64)
print()
print("  Press the key you want to use, a few times, on its own.")
print("  For the ask hotkey on Windows that is usually Right Alt;")
print("  press Right Ctrl too so we can compare against dictation.")
print()
print("  Each press prints its name and scan code. If a key prints")
print("  nothing at all, the hardware hides it and no app can bind it.")
print()
print("  Press ESC three times when you're done.")
print("-" * 64)


def on_event(event):
    global esc_presses, finished
    if finished:
        return

    if event.name == "esc" and event.event_type == "down":
        esc_presses += 1
        if esc_presses >= 3:
            finished = True
            report()
            return
    elif event.event_type == "down":
        esc_presses = 0

    identity = (str(event.name), event.scan_code)
    is_new = identity not in seen
    if is_new:
        seen[identity] = 0
        order.append(identity)
    seen[identity] += 1

    marker = "   <-- NEW" if is_new else ""
    print(f"  {event.event_type:<4}  name={str(event.name):<16} "
          f"scan_code={event.scan_code}{marker}")


def report():
    print()
    print("-" * 64)
    print("  EVERY DISTINCT KEY SEEN (most pressed first)")
    print("-" * 64)
    ranked = sorted(seen.items(), key=lambda kv: -kv[1])
    for (name, code), count in ranked:
        print(f"    name={name!r:<18} scan_code={code:<8} seen {count}x")

    # The key you pressed most, ignoring Esc, is almost certainly the one
    # you want to bind. Suggest a ready-to-paste config for it.
    candidates = [(n, c) for (n, c), _ in ranked if n.lower() not in IGNORE]
    print("-" * 64)
    if candidates:
        name, code = candidates[0]
        print(f"  To bind the key you pressed most ({name!r}), set this in the")
        print(f"  relevant block of config.json (hotkey, or hotkey_ask):")
        print()
        print(f'      "key": null,')
        print(f'      "scan_code": {code}')
        print()
        print(f"  A scan code is exact - it binds this physical key no matter")
        print(f"  what name the driver reports.")
    else:
        print("  No bindable key was captured.")
    print("-" * 64)
    keyboard.unhook_all()


keyboard.hook(on_event)
keyboard.wait()
print("\nProbe finished.")
