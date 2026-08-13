"""
Fn-key probe.

On most laptops the Fn key is handled entirely by the keyboard's embedded
controller: it changes what *other* keys report and never emits a scancode of
its own. When that is the case, no program can bind it. A minority of machines
do send a real code. This prints every raw key event so we can tell which.

Run it, press the keys it asks for, then press ESC three times.
"""

import keyboard

seen: dict[tuple[str, int], int] = {}
esc_presses = 0
finished = False

print("=" * 64)
print("  Fn PROBE")
print("=" * 64)
print()
print("  Press these one at a time, a couple of times each:")
print("    1. Fn            <- the key we actually care about")
print("    2. Right Ctrl    <- the fallback binding")
print("    3. Fn + F1       <- does the combo emit anything?")
print()
print("  If pressing Fn on its own prints no line at all, the")
print("  hardware hides it from Windows and it cannot be bound.")
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
    seen[identity] = seen.get(identity, 0) + 1

    marker = "   <-- NEW" if is_new else ""
    print(f"  {event.event_type:<4}  name={str(event.name):<16} "
          f"scan_code={event.scan_code}{marker}")


def report():
    print()
    print("-" * 64)
    print("  EVERY DISTINCT KEY SEEN")
    print("-" * 64)
    for (name, code), count in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"    name={name!r:<18} scan_code={code:<8} seen {count}x")

    fn_hits = [(n, c) for (n, c) in seen if "fn" in n.lower()]
    print("-" * 64)
    if fn_hits:
        name, code = fn_hits[0]
        print(f"  Fn IS visible to Windows. In config.json set:")
        print(f'      "key": null,')
        print(f'      "scan_code": {code}')
    else:
        print("  No Fn event was ever captured.")
        print("  Your Fn key is firmware-only and cannot be bound by any app.")
        print("  Keep Right Ctrl, or choose another key in config.json.")
    print("-" * 64)
    keyboard.unhook_all()


keyboard.hook(on_event)
keyboard.wait()
print("\nProbe finished.")
