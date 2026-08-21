//! Coverage for the commit PROTOCOL, with no dimension in sight.
//!
//! Every assertion here runs against a binding that is three lines long and means nothing — which
//! is the point. If a test in this file needed a density or a voxel floor to say what it means,
//! the length words would have crept back into the shared half. What a real binding does with the
//! text is asserted next door, in `quantity_binding`.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use super::*;

/// The stand-in binding: it takes any text that is not the word `no`.
///
/// Deliberately not a length. The protocol's rules are about focus, buffers and the seed, and a
/// binding that measured something would let a length rule hide in here.
fn takes_anything(text: &str) -> Result<Accepted<String>, String> {
    if text.trim() == "no" {
        return Err("the binding said no".to_owned());
    }
    Ok(Accepted {
        value: text.trim().to_owned(),
        settled_text: text.trim().to_uppercase(),
    })
}

/// The id the harness below draws its box under.
fn probe_box_id() -> egui::Id {
    egui::Id::new("test_entry").with("probe_box")
}

/// Run one frame of a bare entry, opting into focus or not, and report whether it took the
/// keyboard.
fn a_fresh_entry_takes_focus(opt_in: bool) -> bool {
    let context = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(400.0, 200.0),
        )),
        ..Default::default()
    };
    let _ = context.run_ui(raw_input, |ui| {
        let entry = QuantityEntry::new(egui::Id::new("test_entry"), "seed");
        let entry = if opt_in {
            entry.focus_when_new()
        } else {
            entry
        };
        let _ = entry.run(ui, takes_anything, |ui, buffer| {
            ui.add(egui::TextEdit::singleline(buffer).id(probe_box_id()))
        });
    });
    context.memory(|memory| memory.has_focus(probe_box_id()))
}

/// An entry that opts in takes the keyboard the moment it appears; the default leaves it alone.
///
/// The pair is the whole rule. An inline editor answers a gesture that already said "I mean to
/// change this", so it may take the keyboard; a rail row is present merely because a panel is
/// open, and taking the keyboard there would interrupt whatever the author was actually doing.
#[test]
fn only_an_entry_that_opts_in_takes_the_keyboard_when_it_appears() {
    assert!(
        a_fresh_entry_takes_focus(true),
        "an opened editor is ready to type into"
    );
    assert!(
        !a_fresh_entry_takes_focus(false),
        "a rail row that merely appeared must not steal the keyboard"
    );
}

/// Drive an entry across frames: type `typed` into it (or leave it alone), then take focus away
/// and report what the frame that lost it did.
fn losing_focus_after(typed: Option<&str>) -> QuantityEntryOutcome<String> {
    let context = egui::Context::default();
    let id_base = egui::Id::new("test_entry");
    let screen = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(400.0, 200.0),
        )),
        ..Default::default()
    };

    // Frame one: the box appears and asks for the keyboard.
    let _ = context.run_ui(screen.clone(), |ui| {
        let _ = QuantityEntry::new(id_base, "seed").focus_when_new().run(
            ui,
            takes_anything,
            |ui, buffer| ui.add(egui::TextEdit::singleline(buffer).id(probe_box_id())),
        );
    });

    // Frame two: it has the keyboard, and the author types (or does not).
    let mut typing = screen.clone();
    if let Some(text) = typed {
        typing.events = vec![
            egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            },
            egui::Event::Text(text.to_owned()),
        ];
    }
    let _ = context.run_ui(typing, |ui| {
        let _ = QuantityEntry::new(id_base, "seed").run(ui, takes_anything, |ui, buffer| {
            ui.add(egui::TextEdit::singleline(buffer).id(probe_box_id()))
        });
    });

    // Frame three: Enter, which surrenders focus and is the commit trigger.
    let mut committing = screen;
    committing.events = vec![egui::Event::Key {
        key: egui::Key::Enter,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }];
    let mut outcome = QuantityEntryOutcome::Idle;
    let _ = context.run_ui(committing, |ui| {
        outcome = QuantityEntry::new(id_base, "seed").run(ui, takes_anything, |ui, buffer| {
            ui.add(egui::TextEdit::singleline(buffer).id(probe_box_id()))
        });
    });
    outcome
}

/// Rule 6. A box the author opened and left alone writes NOTHING when focus goes.
///
/// The rule that makes an inline editor safe to open on a gesture: double-clicking a number and
/// then clicking elsewhere has to be a look, not an edit. Without it, opening a box would restate
/// the value it was already showing — which is invisible until the day the seed and the stored
/// value disagree by a rounding, and then it silently moves the drawing.
#[test]
fn an_untouched_seed_commits_nothing() {
    assert_eq!(losing_focus_after(None), QuantityEntryOutcome::Idle);
}

/// And text the author did change commits, settling on whatever the binding said it renders as.
#[test]
fn typed_text_commits_and_settles_on_the_bindings_own_rendering() {
    assert_eq!(
        losing_focus_after(Some("hello")),
        QuantityEntryOutcome::Committed("hello".to_owned())
    );
}

/// A binding that says no leaves the protocol in the refusal state, and the protocol never learns
/// why. That is the refusal channel: one arm, no dimension knowledge.
#[test]
fn a_binding_that_refuses_produces_a_refusal() {
    assert_eq!(
        losing_focus_after(Some("no")),
        QuantityEntryOutcome::Refused
    );
}
