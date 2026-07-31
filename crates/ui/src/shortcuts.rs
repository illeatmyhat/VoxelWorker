//! The keyboard-shortcut settings — **the one place a keybind is written down**.
//!
//! Every command the keyboard can reach is a [`ShortcutCommand`], and each one carries its own
//! built-in binding ([`ShortcutCommand::built_in`]) the way a Krita `<Action>` carries its
//! `<shortcut>` next to its label. [`Shortcuts`] is that inventory plus the user's overrides. A
//! command with `None` is not an omission; it is a command the keyboard cannot reach yet, listed
//! so the settings stay the complete inventory rather than a list of the ones somebody remembered.
//!
//! **Why a registry rather than a literal at each site.** A menu row that spelled its own binding
//! ("Esc", flushed right) and a shell handler that matched its own key are two facts about one
//! binding, free to drift — and the menu is the thing users read to learn the binding, so the copy
//! that drifts is the one that lies. Nothing here can drift: the row is handed a *command* and
//! looks the key up, and the shell asks which commands the frame's presses meant. Neither is
//! offered a string.
//!
//! That is also the enforcement. The shell's `context_menu_row` takes no shortcut text at all, so
//! a hardcoded one is a type error, not a review note. On the winit side the same rule is a clippy
//! `disallowed-types` entry on `KeyCode`, which has no opt-out anywhere: presses are read out of
//! egui's own input, which `egui_winit` has already translated.
//!
//! **What is egui's and what is ours.** The key, the modifiers, the human-readable spelling and
//! the consume-once matching are all [`egui::KeyboardShortcut`] and
//! [`egui::InputState::consume_shortcut`] — including the OS-aware formatting that writes `⌘⇧P` on
//! a Mac and `Ctrl+Shift+P` elsewhere. What egui has no opinion about, and what this module is, is
//! *which commands exist* and *which binding each one holds*.
//!
//! # The shape, and where it comes from
//!
//! Blender and Krita converge on the same four properties, and this module takes all four:
//!
//! 1. **Keyed by command, never by position.** Blender's keymap items name an operator `idname`;
//!    Krita's actions have a `name`. A positional table where row 4 silently means "reset orbit
//!    center" is the thing both avoid.
//! 2. **The default is declared beside the command's own metadata.** Krita puts `<shortcut>` in the
//!    same `<Action>` block as `<text>` and `<toolTip>`; here [`ShortcutCommand::built_in`] sits
//!    next to [`ShortcutCommand::label`], and both platforms' answers for one command are in one
//!    match arm where they can be compared.
//! 3. **The user's changes are stored as a sparse override.** Blender persists a *diff* of
//!    add/remove items against the defaults rather than a copy, so a default that improves reaches
//!    the people who never rebound it. [`Shortcuts`] holds only the overrides, and only those are
//!    persisted.
//! 4. **A whole alternative set is a first-class thing.** Blender ships entire keyconfigs
//!    ("Industry Compatible", Maya); Krita ships shortcut schemes (Photoshop, Paint Tool Sai).
//!    [`ShortcutPlatform`] is that seam here — today it selects the two platform sets, and a
//!    "Fusion-like" or "Blender-like" scheme would enter the same way.
//!
//! # The platform law: each platform's set is written on its own merits
//!
//! The built-in bindings are not one set with a modifier substituted at the edges. Each platform's
//! answer is decided per command, because the platforms disagree about more than which modifier is
//! under the thumb:
//!
//! * The **key itself** can differ. The delete verb is `Delete` on Windows and `Backspace` (⌫) on
//!   a Mac, where the forward-delete key is absent from every laptop keyboard. No modifier rule
//!   produces that.
//! * A key can be **unavailable**. `F9` is Mission Control at the system level on macOS, so a
//!   binding on it is one the app never receives.
//! * The **conventional** shortcut for a verb is sometimes simply a different shortcut, because
//!   that is what people on that platform already have in their fingers.
//!
//! So [`Modifiers::COMMAND`] — egui's "Ctrl here, ⌘ there" modifier — is deliberately **not** used.
//! It is precisely the heuristic this law rejects: it makes the Mac binding a derivative of the
//! Windows one and hides the question of what the Mac binding should be. Each arm names its own
//! platform's real modifiers, and the tests hold it to that.
//!
//! The bindings are **settings** in the ADR 0022 sense: preference that outlives any one project,
//! persisted through a serde mirror out in the shell (this crate links no serde, ADR 0016 — the
//! shortcut type itself is serde-able, so only the command inventory needs mirroring).

use egui::{Key, KeyboardShortcut, Modifiers};
use std::collections::BTreeMap;

/// An unmodified key.
const fn bare(key: Key) -> KeyboardShortcut {
    KeyboardShortcut::new(Modifiers::NONE, key)
}

/// `Ctrl+<key>` — the Windows/Linux spelling, naming `ctrl` because that is the key that is
/// actually pressed there.
const fn ctrl(key: Key) -> KeyboardShortcut {
    KeyboardShortcut::new(
        Modifiers {
            ctrl: true,
            ..Modifiers::NONE
        },
        key,
    )
}

/// `⌘<key>` — the macOS spelling, naming `mac_cmd` for the same reason.
const fn command(key: Key) -> KeyboardShortcut {
    KeyboardShortcut::new(
        Modifiers {
            mac_cmd: true,
            ..Modifiers::NONE
        },
        key,
    )
}

/// `Ctrl+Shift+<key>` — the Windows/Linux spelling, naming `ctrl` because that is the key that is
/// actually pressed there.
const fn ctrl_shift(key: Key) -> KeyboardShortcut {
    KeyboardShortcut::new(
        Modifiers {
            ctrl: true,
            shift: true,
            ..Modifiers::NONE
        },
        key,
    )
}

/// `⌘⇧<key>` — the macOS spelling, naming `mac_cmd` for the same reason.
const fn command_shift(key: Key) -> KeyboardShortcut {
    KeyboardShortcut::new(
        Modifiers {
            mac_cmd: true,
            shift: true,
            ..Modifiers::NONE
        },
        key,
    )
}

/// Which platform's keyboard conventions a set of built-in bindings follows.
///
/// Two variants, not one per OS: Windows and Linux agree with each other about every shortcut in
/// this application, and macOS is the one that does not. This is also the seam an alternative
/// *scheme* would enter through — Blender's keyconfig presets and Krita's shortcut schemes are the
/// same idea, a whole set swapped as a unit rather than a binding patched at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShortcutPlatform {
    /// macOS.
    MacOs,
    /// Windows and Linux, which share these conventions.
    WindowsAndLinux,
}

impl ShortcutPlatform {
    /// The platform this binary was built for. A native app, so the question is settled at compile
    /// time — there is no runtime OS to discover, and reading one would only invite the sets to be
    /// selected by something other than the machine the keys are pressed on.
    pub const HOST: Self = if cfg!(target_os = "macos") {
        Self::MacOs
    } else {
        Self::WindowsAndLinux
    };
}

/// A command the keyboard can be bound to.
///
/// The variants are the inventory the settings list renders, in the order it renders them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShortcutCommand {
    /// End the running modal command, keeping what it produced.
    AcceptCommand,
    /// End the running modal command, discarding what it produced. Also the disarm/back-out key
    /// when no command is running — one binding, a priority chain behind it.
    CancelCommand,
    /// Remove what is picked.
    DeleteSelection,
    /// Reverse the newest edit on the undo stack.
    Undo,
    /// Re-apply the newest undone edit.
    Redo,
    /// Arm the orbit-center placement.
    PlaceOrbitCenter,
    /// Send the orbit center back to the world origin.
    ResetOrbitCenter,
    /// Enter the explicit orbit mode, naming the constrained type.
    EnterConstrainedOrbit,
    /// Carve or fill the sketch region under the cursor.
    ToggleSketchFace,
    /// Dump the live scene + camera to the repro file.
    ExportRepro,
}

impl ShortcutCommand {
    /// Every command, in settings-list order.
    pub const ALL: [Self; 10] = [
        Self::AcceptCommand,
        Self::CancelCommand,
        Self::DeleteSelection,
        Self::Undo,
        Self::Redo,
        Self::PlaceOrbitCenter,
        Self::ResetOrbitCenter,
        Self::EnterConstrainedOrbit,
        Self::ToggleSketchFace,
        Self::ExportRepro,
    ];

    /// The command's name in the settings list.
    pub fn label(self) -> &'static str {
        match self {
            Self::AcceptCommand => "Accept command",
            Self::CancelCommand => "Cancel command",
            Self::DeleteSelection => "Delete selection",
            Self::Undo => "Undo",
            Self::Redo => "Redo",
            Self::PlaceOrbitCenter => "Place orbit center",
            Self::ResetOrbitCenter => "Reset orbit center",
            Self::EnterConstrainedOrbit => "Constrained orbit",
            Self::ToggleSketchFace => "Carve / fill sketch region",
            Self::ExportRepro => "Dump repro",
        }
    }

    /// The command's built-in binding on `platform`, before any user override.
    ///
    /// One arm per command with both platforms inside it, so "was this decided independently, or
    /// copied?" is answerable by reading the arm. Most commands answer the same on both — that is
    /// a conclusion (confirm really is Return everywhere), not a default.
    pub const fn built_in(self, platform: ShortcutPlatform) -> Option<KeyboardShortcut> {
        match self {
            // The universal pair. Return confirms and Escape backs out on every platform this
            // application runs on; there is nothing to decide differently.
            Self::AcceptCommand => Some(bare(Key::Enter)),
            Self::CancelCommand => Some(bare(Key::Escape)),

            // The delete verb, and the case the platform law exists for: the KEY differs, not the
            // modifier. Windows keyboards have a forward-delete and that is the one people press;
            // no Mac laptop has ever had one, so ⌫ is the delete key there and Delete would be a
            // binding most Mac users cannot reach.
            Self::DeleteSelection => match platform {
                ShortcutPlatform::WindowsAndLinux => Some(bare(Key::Delete)),
                ShortcutPlatform::MacOs => Some(bare(Key::Backspace)),
            },

            // Undo agrees across platforms up to the application modifier; Redo is the case the
            // platform law exists for again — Windows' convention (and Fusion's) is `Ctrl+Y`, the
            // Mac's is `⌘⇧Z`, and neither is a modifier-substitution of the other.
            Self::Undo => match platform {
                ShortcutPlatform::WindowsAndLinux => Some(ctrl(Key::Z)),
                ShortcutPlatform::MacOs => Some(command(Key::Z)),
            },
            Self::Redo => match platform {
                ShortcutPlatform::WindowsAndLinux => Some(ctrl(Key::Y)),
                ShortcutPlatform::MacOs => Some(command_shift(Key::Z)),
            },

            // Unbound on both. These are viewport verbs with no cross-application convention, and
            // one that claimed a letter key would be taking it from every future mode at once.
            // They are listed anyway so they can BE bound — the inventory is the point.
            Self::PlaceOrbitCenter
            | Self::ResetOrbitCenter
            | Self::EnterConstrainedOrbit
            | Self::ToggleSketchFace => None,

            // The repro dump: a developer affordance, so it wants a chord nothing else claims
            // rather than a key a modeller might hit by accident. `Shift` plus the platform's own
            // application modifier, on a letter no viewport verb wants.
            Self::ExportRepro => match platform {
                ShortcutPlatform::WindowsAndLinux => Some(ctrl_shift(Key::P)),
                ShortcutPlatform::MacOs => Some(command_shift(Key::P)),
            },
        }
    }
}

/// How many modifiers a shortcut carries — its **specificity**.
///
/// egui matches modifiers logically, so `Ctrl+Shift+S` would also satisfy a bare `Ctrl+S` check.
/// Consuming the more specific binding first is what stops the plainer one from stealing the
/// press, and is why [`Shortcuts::consume`] sorts by this.
fn specificity(shortcut: KeyboardShortcut) -> u32 {
    let Modifiers {
        alt,
        ctrl,
        shift,
        mac_cmd,
        command,
    } = shortcut.modifiers;
    u32::from(alt) + u32::from(ctrl) + u32::from(shift) + u32::from(mac_cmd) + u32::from(command)
}

/// The keyboard-shortcut settings: a platform's built-in bindings plus the user's overrides.
///
/// Only the overrides are stored, Blender-style. Holding a full copy of the table would freeze
/// today's defaults into every existing config, so a binding improved next year would reach only
/// people who had never opened the settings.
#[derive(Debug, Clone, PartialEq)]
pub struct Shortcuts {
    /// Whose built-in set the overrides sit on top of.
    platform: ShortcutPlatform,
    /// The commands the user has changed. The value is the *new* binding: `None` means explicitly
    /// unbound, which is different from "not overridden" (absent from the map).
    overrides: BTreeMap<ShortcutCommand, Option<KeyboardShortcut>>,
}

impl Shortcuts {
    /// A platform's built-in set, with nothing overridden.
    pub fn for_platform(platform: ShortcutPlatform) -> Self {
        Self {
            platform,
            overrides: BTreeMap::new(),
        }
    }

    /// The binding in force for `command`: its override if it has one, else the built-in.
    pub fn shortcut(&self, command: ShortcutCommand) -> Option<KeyboardShortcut> {
        match self.overrides.get(&command) {
            Some(overridden) => *overridden,
            None => command.built_in(self.platform),
        }
    }

    /// How `command`'s binding is spelled on screen, if it has one — egui's own OS-aware
    /// formatting, so the menu column and the settings list read the same and read native.
    pub fn display(&self, ctx: &egui::Context, command: ShortcutCommand) -> Option<String> {
        self.shortcut(command)
            .map(|shortcut| ctx.format_shortcut(&shortcut))
    }

    /// Override `command`'s binding, taking the shortcut off whatever else held it — one chord
    /// means one command, so rebinding never leaves two handlers racing for the same press.
    pub fn bind(&mut self, command: ShortcutCommand, shortcut: Option<KeyboardShortcut>) {
        if let Some(shortcut) = shortcut {
            for held_by in ShortcutCommand::ALL {
                if held_by != command && self.shortcut(held_by) == Some(shortcut) {
                    self.overrides.insert(held_by, None);
                }
            }
        }
        self.overrides.insert(command, shortcut);
    }

    /// Drop `command`'s override, returning it to the built-in binding.
    pub fn reset(&mut self, command: ShortcutCommand) {
        self.overrides.remove(&command);
    }

    /// Which command holds `shortcut`, if any.
    pub fn command(&self, shortcut: KeyboardShortcut) -> Option<ShortcutCommand> {
        ShortcutCommand::ALL
            .into_iter()
            .find(|command| self.shortcut(*command) == Some(shortcut))
    }

    /// The commands whose bindings were pressed this frame, **consuming** the presses.
    ///
    /// Call it AFTER the egui pass: a focused text field has already eaten its keys by then, so a
    /// typed Escape ends the edit instead of canceling the running viewport command. That
    /// ordering is the reason this reads egui's input rather than the raw winit event — the guard
    /// is structural instead of a "was egui focused?" flag the caller has to remember.
    pub fn consume(&self, ctx: &egui::Context) -> Vec<ShortcutCommand> {
        let mut bound: Vec<(ShortcutCommand, KeyboardShortcut)> = ShortcutCommand::ALL
            .into_iter()
            .filter_map(|command| self.shortcut(command).map(|shortcut| (command, shortcut)))
            .collect();
        bound.sort_by_key(|(_, shortcut)| std::cmp::Reverse(specificity(*shortcut)));
        ctx.input_mut(|input| {
            bound
                .into_iter()
                .filter(|(_, shortcut)| input.consume_shortcut(shortcut))
                .map(|(command, _)| command)
                .collect()
        })
    }

    /// The settings list: every command with the binding in force, in inventory order.
    pub fn list(&self) -> impl Iterator<Item = (ShortcutCommand, Option<KeyboardShortcut>)> + '_ {
        ShortcutCommand::ALL
            .into_iter()
            .map(|command| (command, self.shortcut(command)))
    }

    /// Just the overrides — what persistence writes down, and all it writes down.
    pub fn overrides(
        &self,
    ) -> impl Iterator<Item = (ShortcutCommand, Option<KeyboardShortcut>)> + '_ {
        self.overrides
            .iter()
            .map(|(command, shortcut)| (*command, *shortcut))
    }
}

impl Default for Shortcuts {
    fn default() -> Self {
        Self::for_platform(ShortcutPlatform::HOST)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each platform's set names only modifiers its own platform has. A `ctrl` on the Mac side is
    /// a binding under the wrong finger; a `mac_cmd` on the other is one that key cannot press.
    #[test]
    fn each_platform_names_only_its_own_modifiers() {
        for command in ShortcutCommand::ALL {
            if let Some(shortcut) = command.built_in(ShortcutPlatform::MacOs) {
                assert!(
                    !shortcut.modifiers.ctrl,
                    "the Mac binding for {command:?} chords with Ctrl"
                );
            }
            if let Some(shortcut) = command.built_in(ShortcutPlatform::WindowsAndLinux) {
                assert!(
                    !shortcut.modifiers.mac_cmd,
                    "the Windows/Linux binding for {command:?} chords with ⌘"
                );
            }
        }
    }

    /// The platform law itself. `Modifiers::COMMAND` is egui's "Ctrl here, ⌘ there" — the exact
    /// substitution heuristic the per-command arms exist to replace. A binding that reaches for it
    /// has stopped asking what the shortcut should BE on each platform and started deriving one
    /// from the other.
    #[test]
    fn no_binding_is_derived_from_the_other_platforms() {
        for platform in [ShortcutPlatform::MacOs, ShortcutPlatform::WindowsAndLinux] {
            for command in ShortcutCommand::ALL {
                let Some(shortcut) = command.built_in(platform) else {
                    continue;
                };
                assert!(
                    !shortcut.modifiers.command,
                    "{platform:?}'s {command:?} uses Modifiers::COMMAND; name the platform's own \
                     modifier and decide the binding on its merits"
                );
            }
        }
    }

    /// Every platform's set is a valid registry in its own right, not only the host's.
    #[test]
    fn no_platform_binds_one_shortcut_to_two_commands() {
        for platform in [ShortcutPlatform::MacOs, ShortcutPlatform::WindowsAndLinux] {
            let shortcuts = Shortcuts::for_platform(platform);
            for (command, shortcut) in shortcuts.list() {
                let Some(shortcut) = shortcut else { continue };
                assert_eq!(shortcuts.command(shortcut), Some(command), "{platform:?}");
            }
        }
    }

    #[test]
    fn binding_a_held_shortcut_takes_it_off_the_previous_command() {
        let escape = bare(Key::Escape);
        let mut shortcuts = Shortcuts::default();
        shortcuts.bind(ShortcutCommand::DeleteSelection, Some(escape));
        assert_eq!(shortcuts.shortcut(ShortcutCommand::CancelCommand), None);
        assert_eq!(
            shortcuts.command(escape),
            Some(ShortcutCommand::DeleteSelection)
        );
    }

    /// An explicit unbind is an override, not an absence — otherwise it would read as "never
    /// touched" and the built-in would come straight back.
    #[test]
    fn an_explicit_unbind_survives_and_reset_undoes_it() {
        let mut shortcuts = Shortcuts::default();
        shortcuts.bind(ShortcutCommand::CancelCommand, None);
        assert_eq!(shortcuts.shortcut(ShortcutCommand::CancelCommand), None);
        assert_eq!(shortcuts.overrides().count(), 1);
        shortcuts.reset(ShortcutCommand::CancelCommand);
        assert_eq!(
            shortcuts.shortcut(ShortcutCommand::CancelCommand),
            Some(bare(Key::Escape))
        );
        assert_eq!(shortcuts.overrides().count(), 0);
    }

    /// Only what the user changed is written down, so a default improved later still reaches them.
    #[test]
    fn an_untouched_set_has_nothing_to_persist() {
        assert_eq!(Shortcuts::default().overrides().count(), 0);
    }

    #[test]
    fn a_chord_outranks_the_bare_key_it_contains() {
        assert!(specificity(ctrl_shift(Key::P)) > specificity(bare(Key::P)));
    }

    #[test]
    fn the_list_covers_every_command() {
        assert_eq!(
            Shortcuts::default().list().count(),
            ShortcutCommand::ALL.len()
        );
    }
}
