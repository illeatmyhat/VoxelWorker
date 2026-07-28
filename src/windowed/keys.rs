//! The winit → [`ShortcutKey`] translation table — **the only place a `KeyCode` is named**.
//!
//! Everything else in the shell asks the [`Shortcuts`](ui::shortcuts::Shortcuts) settings which
//! *command* a press means, so a rebound key moves one entry in the settings instead of chasing
//! `KeyCode::` literals through the event loop. The clippy `disallowed-types` entry on `KeyCode`
//! is what keeps that true; this module is the one opt-out.

// The one opt-out from the lint that keeps `KeyCode` out of the rest of the shell.
#![allow(clippy::disallowed_types)]

use ui::shortcuts::ShortcutKey;

/// The bindable key a winit press denotes, if it is one.
///
/// Unbindable keys are `None` rather than a catch-all variant: a key with no [`ShortcutKey`] is a
/// key no command can be bound to, which is exactly what the settings picker should not offer.
pub(super) fn shortcut_key(code: winit::keyboard::KeyCode) -> Option<ShortcutKey> {
    use winit::keyboard::KeyCode;
    match code {
        // NumpadEnter is the same verb under a different finger; the settings list has one Return.
        KeyCode::Enter | KeyCode::NumpadEnter => Some(ShortcutKey::Return),
        KeyCode::Escape => Some(ShortcutKey::Escape),
        KeyCode::Delete => Some(ShortcutKey::Delete),
        KeyCode::Backspace => Some(ShortcutKey::Backspace),
        KeyCode::F9 => Some(ShortcutKey::F9),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bindable_key_has_a_winit_press_that_produces_it() {
        for key in ShortcutKey::ALL {
            let reachable = [
                winit::keyboard::KeyCode::Enter,
                winit::keyboard::KeyCode::NumpadEnter,
                winit::keyboard::KeyCode::Escape,
                winit::keyboard::KeyCode::Delete,
                winit::keyboard::KeyCode::Backspace,
                winit::keyboard::KeyCode::F9,
            ]
            .into_iter()
            .any(|code| shortcut_key(code) == Some(key));
            assert!(
                reachable,
                "{key:?} is bindable but no key press produces it"
            );
        }
    }
}
