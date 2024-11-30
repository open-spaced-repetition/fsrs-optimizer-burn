use fsrs::{FSRS, FSRSItem, FSRSReview};

fn migrate_with_full_history() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new FSRS model
    let fsrs = FSRS::new(Some(&[]))?;

    // Simulate a full review history for a card
    let reviews = vec![
        FSRSReview {
            rating: 3,
            delta_t: 0,
        },
        FSRSReview {
            rating: 3,
            delta_t: 1,
        },
        FSRSReview {
            rating: 4,
            delta_t: 3,
        },
        FSRSReview {
            rating: 3,
            delta_t: 7,
        },
    ];
    let item = FSRSItem { reviews };

    // Calculate the current memory state
    let memory_state = fsrs.memory_state(item, None)?;

    println!("Migrated memory state: {:?}", memory_state);

    Ok(())
}

fn migrate_with_partial_history() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new FSRS model
    let fsrs = FSRS::new(Some(&[]))?;

    // Set the true retention of the original algorithm
    let sm2_retention = 0.9;

    // Simulate the earliest card state from the first review log of Anki's card
    // - ease_factor: the ratio of the interval to the previous interval
    // - interval: the interval of the first review
    let ease_factor = 2.0;
    let interval = 5.0;

    // Calculate the earliest memory state
    let initial_state = fsrs.memory_state_from_sm2(ease_factor, interval, sm2_retention)?;

    // Simulate partial review history
    let reviews = vec![
        FSRSReview {
            rating: 3,
            delta_t: 5,
        },
        FSRSReview {
            rating: 4,
            delta_t: 10,
        },
        FSRSReview {
            rating: 3,
            delta_t: 20,
        },
    ];
    let item = FSRSItem { reviews };

    // Calculate the current memory state, passing the initial state
    let memory_state = fsrs.memory_state(item, Some(initial_state))?;

    println!("Migrated memory state: {:?}", memory_state);

    Ok(())
}

fn migrate_with_latest_state() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new FSRS model
    let fsrs = FSRS::new(Some(&[]))?;

    // Set the true retention of the original algorithm
    let sm2_retention = 0.9;

    // Simulate the latest card state from Anki's card
    // - ease_factor: the ratio of the interval to the previous interval
    // - interval: the interval of the last review
    let ease_factor = 2.5;
    let interval = 10.0;

    // Calculate the memory state
    let memory_state = fsrs.memory_state_from_sm2(ease_factor, interval, sm2_retention)?;

    println!("Migrated memory state: {:?}", memory_state);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Migrating with full history:");
    migrate_with_full_history()?;

    println!("\nMigrating with partial history:");
    migrate_with_partial_history()?;

    println!("\nMigrating with latest state only:");
    migrate_with_latest_state()?;

    Ok(())
}
