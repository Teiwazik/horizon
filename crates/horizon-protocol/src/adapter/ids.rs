//! Registry ids an older server numbers differently: the tables `tools/generate_registry_id_adapter.py`
//! writes, and the per-thread context the item and particle readers consult while a packet from such a
//! server is decoded (an id there decides how the bytes after it read, so it must be renumbered as it is
//! read, not afterwards).
use std::cell::Cell;

/// How one registry's ids differ between an older version and the pinned one.
#[derive(Clone, Copy, Debug)]
pub enum Remap {
    /// Numbered alike.
    Same,
    /// Runs `(from, to, length)`: ids `from..from + length` map to `to..to + length`. `up` goes from the older
    /// version to the pinned one, `down` back. An id in no run has no entry on the other side.
    Runs {
        up: &'static [(u16, u16, u16)],
        down: &'static [(u16, u16, u16)],
    },
}

fn lookup(runs: &[(u16, u16, u16)], id: i32) -> Option<i32> {
    let id = u16::try_from(id).ok()?;
    let i = runs
        .partition_point(|&(from, _, _)| from <= id)
        .checked_sub(1)?;
    let (from, to, length) = runs[i];
    (id - from < length).then(|| i32::from(to + (id - from)))
}

impl Remap {
    /// The pinned id of the older version's `id`; `None` when the pinned version has no such entry.
    pub fn up(self, id: i32) -> Option<i32> {
        match self {
            Self::Same => Some(id),
            Self::Runs { up, .. } => lookup(up, id),
        }
    }
    /// The older version's id of the pinned `id`; `None` when the older version has no such entry.
    pub fn down(self, id: i32) -> Option<i32> {
        match self {
            Self::Same => Some(id),
            Self::Runs { down, .. } => lookup(down, id),
        }
    }
}

/// Block states and every registry the protocol sends by id, for one older version.
#[derive(Clone, Copy, Debug)]
pub struct Ids {
    /// How many block states the older version and the pinned one have.
    pub states: (u32, u32),
    pub block_state: Remap,
    pub item: Remap,
    pub entity_type: Remap,
    pub particle_type: Remap,
    pub sound_event: Remap,
    pub data_component_type: Remap,
    pub command_argument_type: Remap,
    pub custom_stat: Remap,
    pub map_decoration_type: Remap,
    pub block: Remap,
    pub block_entity_type: Remap,
    pub menu: Remap,
    pub mob_effect: Remap,
    pub attribute: Remap,
    pub stat_type: Remap,
    pub fluid: Remap,
    pub game_event: Remap,
    pub recipe_serializer: Remap,
    pub slot_display: Remap,
    pub recipe_display: Remap,
    pub recipe_book_category: Remap,
    pub villager_type: Remap,
    pub villager_profession: Remap,
    pub position_source_type: Remap,
    pub consume_effect_type: Remap,
    pub enchantment_effect_component_type: Remap,
}

impl Ids {
    /// Bits per entry of a chunk section's global block palette: the older version's, then the pinned one's
    /// (`Mth.ceillog2` of the state count).
    pub fn state_bits(&self) -> (u8, u8) {
        let bits = |n: u32| (u32::BITS - n.saturating_sub(1).leading_zeros()) as u8;
        (bits(self.states.0), bits(self.states.1))
    }
}

/// Which way ids are being renumbered on this thread.
#[derive(Clone, Copy, Debug)]
pub enum Way {
    /// Decoding what an older server sent: its ids become the pinned version's.
    Up,
    /// Encoding for an older server: the pinned ids become its own.
    Down,
}

thread_local! {
    static ACTIVE: Cell<Option<(&'static Ids, Way)>> = const { Cell::new(None) };
}

/// Renumber ids `way` on this thread until the guard drops. Readers and writers that pick a layout by an id
/// (item stacks, data components, particles) consult it through [`item`], [`component`] and [`particle`].
pub fn renumbering(ids: &'static Ids, way: Way) -> Guard {
    Guard(ACTIVE.replace(Some((ids, way))))
}

/// Restores the previous context when dropped.
pub struct Guard(Option<(&'static Ids, Way)>);
impl Drop for Guard {
    fn drop(&mut self) {
        ACTIVE.set(self.0);
    }
}

fn apply(id: i32, pick: impl Fn(&Ids) -> Remap) -> i32 {
    match ACTIVE.get() {
        None => id,
        // An id the other side lacks becomes -1, which every reader turns away or reads as nothing.
        Some((ids, Way::Up)) => pick(ids).up(id).unwrap_or(-1),
        Some((ids, Way::Down)) => pick(ids).down(id).unwrap_or(-1),
    }
}

/// An item id as read or about to be written on this thread.
pub fn item(id: i32) -> i32 {
    apply(id, |ids| ids.item)
}
/// A data component type id as read or about to be written on this thread.
pub fn component(id: i32) -> i32 {
    apply(id, |ids| ids.data_component_type)
}
/// A particle type id as read or about to be written on this thread.
pub fn particle(id: i32) -> i32 {
    apply(id, |ids| ids.particle_type)
}
/// A block state id as read or about to be written on this thread.
pub fn block_state(id: i32) -> i32 {
    apply(id, |ids| ids.block_state)
}
/// An entity type id as read or about to be written on this thread.
pub fn entity_type(id: i32) -> i32 {
    apply(id, |ids| ids.entity_type)
}
/// A sound event id as read or about to be written on this thread.
pub fn sound_event(id: i32) -> i32 {
    apply(id, |ids| ids.sound_event)
}
/// The ids on this thread now, if any.
pub fn active() -> Option<(&'static Ids, Way)> {
    ACTIVE.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_map_inside_and_nothing_outside() {
        let runs = Remap::Runs {
            up: &[(0, 0, 3), (3, 5, 2)],
            down: &[(0, 0, 3), (5, 3, 2)],
        };
        assert_eq!(runs.up(2), Some(2));
        assert_eq!(runs.up(3), Some(5));
        assert_eq!(runs.up(4), Some(6));
        assert_eq!(runs.up(5), None);
        assert_eq!(runs.up(-1), None);
        assert_eq!(runs.down(3), None);
        assert_eq!(runs.down(6), Some(4));
    }

    #[test]
    fn the_context_lasts_while_its_guard_does() {
        static IDS: Ids = super::super::v26_2_ids::IDS;
        assert_eq!(item(926), 926);
        {
            // diamond: 926 in 26.2, 1012 in 26.3.
            let _up = renumbering(&IDS, Way::Up);
            assert_eq!(item(926), 1012);
            {
                let _down = renumbering(&IDS, Way::Down);
                assert_eq!(item(1012), 926);
            }
            assert_eq!(item(926), 1012);
        }
        assert_eq!(item(926), 926);
    }
}
