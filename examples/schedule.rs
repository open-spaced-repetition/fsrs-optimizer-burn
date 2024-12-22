use chrono::{DateTime, Duration, Utc};
use fsrs::{FSRS, MemoryState};

struct Card {
    due: DateTime<Utc>,
    memory_state: Option<MemoryState>,
    scheduled_days: u32,
    last_review: Option<DateTime<Utc>>,
}

impl Card {
    pub fn new() -> Self {
        Self {
            due: Utc::now(),
            last_review: None,
            memory_state: None,
            scheduled_days: 0,
        }
    }
}

fn schedule_new_card() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new card
    let mut card = Card::new();

    // Set desired retention
    let desired_retention = 0.9;

    // Create a new FSRS model
    let fsrs = FSRS::new(Some(&[]))?;

    // Get next states for a new card
    let next_states = fsrs.next_states(card.memory_state, desired_retention, 0)?;

    // Display the intervals for each rating
    println!(
        "Again interval: {} days",
        next_states.again.interval.round().max(1.0)
    );
    println!(
        "Hard interval: {} days",
        next_states.hard.interval.round().max(1.0)
    );
    println!(
        "Good interval: {} days",
        next_states.good.interval.round().max(1.0)
    );
    println!(
        "Easy interval: {} days",
        next_states.easy.interval.round().max(1.0)
    );

    // Assume the card was reviewed and the rating was `good`
    let next_state = next_states.good;
    let interval = next_state.interval.round().max(1.0) as u32;

    // Update the card with the new memory state and interval
    card.memory_state = Some(next_state.memory);
    card.scheduled_days = interval;
    card.last_review = Some(Utc::now());
    card.due = card.last_review.unwrap() + Duration::days(interval as i64);

    println!("Next review due: {}", card.due);
    println!("Memory state: {:?}", card.memory_state);
    Ok(())
}

fn schedule_existing_card() -> Result<(), Box<dyn std::error::Error>> {
    // Create an existing card with memory state and last review date
    let mut card = Card {
        due: Utc::now(),                                   // Due today
        last_review: Some(Utc::now() - Duration::days(7)), // Last reviewed 7 days ago
        memory_state: Some(MemoryState {
            stability: 7.0,
            difficulty: 5.0,
        }),
        scheduled_days: 7,
    };

    // Set desired retention
    let desired_retention = 0.9;

    // Create a new FSRS model
    let fsrs = FSRS::new(Some(&[]))?;

    // Calculate the elapsed time since the last review
    let elapsed_days = (Utc::now() - card.last_review.unwrap()).num_days() as u32;

    // Get next states for an existing card
    let next_states = fsrs.next_states(card.memory_state, desired_retention, elapsed_days)?;

    // Display the intervals for each rating
    println!(
        "Again interval: {} days",
        next_states.again.interval.round().max(1.0)
    );
    println!(
        "Hard interval: {} days",
        next_states.hard.interval.round().max(1.0)
    );
    println!(
        "Good interval: {} days",
        next_states.good.interval.round().max(1.0)
    );
    println!(
        "Easy interval: {} days",
        next_states.easy.interval.round().max(1.0)
    );

    // Assume the card was reviewed and the rating was `again`
    let next_state = next_states.again;
    let interval = next_state.interval.round().max(1.0) as u32;

    // Update the card with the new memory state and interval
    card.memory_state = Some(next_state.memory);
    card.scheduled_days = interval;
    card.last_review = Some(Utc::now());
    card.due = card.last_review.unwrap() + Duration::days(interval as i64);

    println!("Next review due: {}", card.due);
    println!("Memory state: {:?}", card.memory_state);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Scheduling a new card:");
    schedule_new_card()?;

    println!("\nScheduling an existing card:");
    schedule_existing_card()?;

    Ok(())
}
