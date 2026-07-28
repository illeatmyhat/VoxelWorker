//! The keyboard-shortcut settings — **the one place a keybind is written down**.
//!
//! Every command the keyboard can reach is a [`ShortcutCommand`], and [`Shortcuts`] maps each to
//! at most one [`ShortcutKey`]. A command with `None` is not an omission; it is a command the
//! keyboard cannot reach yet, listed so the settings stay the complete inventory rather than a
//! list of the ones somebody remembered.
//!
//! **Why a registry rather than a literal at each site.** A menu row that spelled its own binding
//! ("Esc", flushed right) and a shell handler that matched its own key are two facts about one
//! binding, free to drift — and the menu is the thing users read to learn the binding, so the copy
//! that drifts is the one that lies. Nothing here can drift: the row is handed a *command* and
//! looks the key up, and the shell asks which command a key means. Neither is offered a string.
//!
//! That is also the enforcement. The shell's `context_menu_row` takes no shortcut text at all, so
//! a hardcoded one is a type error, not a review note. On the winit side the same rule is a clippy
//! `disallowed-types` entry on `KeyCode`, with the single translation table opting out.
//!
//! The bindings are **settings** in the ADR 0022 sense: preference that outlives any one project,
//! persisted through a serde mirror out in the shell (this crate links no serde, ADR 0016).

/// A command the keyboard can be bound to.
///
/// The variants are the inventory the settings list renders, in the order it renders them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShortcutCommand {
    /// End the running modal command, keeping what it produced.
    AcceptCommand,
    /// End the running modal command, discarding what it produced. Also the disarm/back-out key
    /// when no command is running — one binding, a priority chain behind it.
    CancelCommand,
    /// Remove what is picked.
    DeleteSelection,
    /// Arm the orbit-center placement.
    PlaceOrbitCenter,
    /// Send the orbit center back to the world origin.
    ResetOrbitCenter,
    /// Enter the explicit orbit mode, naming the constrained type.
    EnterConstrainedOrbit,
    /// Dump the live scene + camera to the repro file.
    ExportRepro,
}

impl ShortcutCommand {
    /// Every command, in settings-list order. The array length is the registry's width, so a new
    /// variant that is not added here fails to compile at [`Shortcuts::DEFAULT`].
    pub const ALL: [Self; 7] = [
        Self::AcceptCommand,
        Self::CancelCommand,
        Self::DeleteSelection,
        Self::PlaceOrbitCenter,
        Self::ResetOrbitCenter,
        Self::EnterConstrainedOrbit,
        Self::ExportRepro,
    ];

    /// The command's name in the settings list.
    pub fn label(self) -> &'static str {
        match self {
            Self::AcceptCommand => "Accept command",
            Self::CancelCommand => "Cancel command",
            Self::DeleteSelection => "Delete selection",
            Self::PlaceOrbitCenter => "Place orbit center",
            Self::ResetOrbitCenter => "Reset orbit center",
            Self::EnterConstrainedOrbit => "Constrained orbit",
            Self::ExportRepro => "Dump repro",
        }
    }

    /// Position in [`ALL`](Self::ALL) — the registry's storage index.
    fn index(self) -> usize {
        match self {
            Self::AcceptCommand => 0,
            Self::CancelCommand => 1,
            Self::DeleteSelection => 2,
            Self::PlaceOrbitCenter => 3,
            Self::ResetOrbitCenter => 4,
            Self::EnterConstrainedOrbit => 5,
            Self::ExportRepro => 6,
        }
    }
}

/// A bindable key.
///
/// Bare keys only — no modifiers, because no command wants one yet and a chord type nobody uses
/// would be a second thing to keep in step with the translation table. Adding modifiers means
/// widening this type, which is exactly the edit that should be visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShortcutKey {
    /// The Return / Enter key.
    Return,
    /// The Escape key.
    Escape,
    /// The forward-delete key.
    Delete,
    /// The backspace key.
    Backspace,
    /// The F9 function key.
    F9,
}

impl ShortcutKey {
    /// Every key that can be bound, for the settings list's picker.
    pub const ALL: [Self; 5] = [
        Self::Return,
        Self::Escape,
        Self::Delete,
        Self::Backspace,
        Self::F9,
    ];

    /// How the key is spelled on screen — in the menu's right-hand column and in the settings
    /// list, from this one definition.
    pub fn display(self) -> &'static str {
        match self {
            Self::Return => "Return",
            Self::Escape => "Esc",
            Self::Delete => "Del",
            Self::Backspace => "Backspace",
            Self::F9 => "F9",
        }
    }
}

/// The keyboard-shortcut settings: one optional key per command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shortcuts {
    /// Indexed by [`ShortcutCommand::index`]; `None` is "not reachable from the keyboard".
    bindings: [Option<ShortcutKey>; ShortcutCommand::ALL.len()],
}

impl Shortcuts {
    /// The built-in bindings. Only the universal pair is bound out of the box — a viewport verb
    /// that took a letter key would be claiming it from every future mode at once.
    pub const DEFAULT: Self = Self {
        bindings: [
            Some(ShortcutKey::Return),
            Some(ShortcutKey::Escape),
            None,
            None,
            None,
            None,
            Some(ShortcutKey::F9),
        ],
    };

    /// The key bound to `command`, if any.
    pub fn key(&self, command: ShortcutCommand) -> Option<ShortcutKey> {
        self.bindings[command.index()]
    }

    /// How `command`'s binding is spelled on screen, if it has one.
    pub fn display(&self, command: ShortcutCommand) -> Option<&'static str> {
        self.key(command).map(ShortcutKey::display)
    }

    /// Bind `command` to `key`, clearing that key off whatever else held it — one key means one
    /// command, so rebinding never leaves two handlers racing for the same press.
    pub fn bind(&mut self, command: ShortcutCommand, key: Option<ShortcutKey>) {
        if let Some(key) = key {
            for binding in &mut self.bindings {
                if *binding == Some(key) {
                    *binding = None;
                }
            }
        }
        self.bindings[command.index()] = key;
    }

    /// Which command a pressed key means, if any. The shell's whole key dispatch.
    pub fn command(&self, key: ShortcutKey) -> Option<ShortcutCommand> {
        ShortcutCommand::ALL
            .into_iter()
            .find(|command| self.key(*command) == Some(key))
    }

    /// The settings list: every command with its binding, in inventory order.
    pub fn list(&self) -> impl Iterator<Item = (ShortcutCommand, Option<ShortcutKey>)> + '_ {
        ShortcutCommand::ALL
            .into_iter()
            .map(|command| (command, self.key(command)))
    }
}

impl Default for Shortcuts {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_index_agrees_with_the_inventory_order() {
        for (position, command) in ShortcutCommand::ALL.into_iter().enumerate() {
            assert_eq!(command.index(), position, "{command:?}");
        }
    }

    #[test]
    fn no_key_is_bound_to_two_commands() {
        let shortcuts = Shortcuts::default();
        for key in ShortcutKey::ALL {
            let holders = ShortcutCommand::ALL
                .into_iter()
                .filter(|command| shortcuts.key(*command) == Some(key))
                .count();
            assert!(holders <= 1, "{key:?} is bound to {holders} commands");
        }
    }

    #[test]
    fn binding_a_held_key_takes_it_off_the_previous_command() {
        let mut shortcuts = Shortcuts::default();
        shortcuts.bind(ShortcutCommand::DeleteSelection, Some(ShortcutKey::Escape));
        assert_eq!(shortcuts.key(ShortcutCommand::CancelCommand), None);
        assert_eq!(
            shortcuts.command(ShortcutKey::Escape),
            Some(ShortcutCommand::DeleteSelection)
        );
    }

    #[test]
    fn the_list_covers_every_command() {
        assert_eq!(
            Shortcuts::default().list().count(),
            ShortcutCommand::ALL.len()
        );
    }
}
