use chrono::NaiveDate;
use fsrs::{DEFAULT_PARAMETERS, FSRS, FSRSItem, FSRSReview};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create review histories for cards
    let review_histories_of_cards = create_review_histories_for_cards();

    // Convert review histories to FSRSItems
    let fsrs_items: Vec<FSRSItem> = review_histories_of_cards
        .into_iter()
        .flat_map(convert_to_fsrs_item)
        .collect();

    println!("Size of FSRSItems: {}", fsrs_items.len());

    // Create an FSRS instance with default parameters
    let fsrs = FSRS::new(Some(&[]))?;
    println!("Default parameters: {:?}", DEFAULT_PARAMETERS);

    // Optimize the FSRS model using the created items
    let optimized_parameters = fsrs.compute_parameters(fsrs_items, None)?;

    println!("Optimized parameters: {:?}", optimized_parameters);

    Ok(())
}

fn create_review_histories_for_cards() -> Vec<Vec<(NaiveDate, u32)>> {
    // This vector represents a collection of review histories for multiple cards.
    // Each inner vector represents the review history of a single card.
    // The structure is as follows:
    // - Outer vector: Contains review histories for multiple cards
    // - Inner vector: Represents the review history of a single card
    //   - Each element is a tuple: (Date, Rating)
    //     - Date: The date of the review (NaiveDate)
    //     - Rating: The rating given during the review (u32)
    //
    // The ratings typically follow this scale:
    // 1: Again, 2: Hard, 3: Good, 4: Easy
    //
    // This sample data includes various review patterns, such as:
    // - Cards with different numbers of reviews
    // - Various intervals between reviews
    // - Different rating patterns (e.g., consistently high, mixed, or improving over time)
    //
    // The data is then cycled and repeated to create a larger dataset of 100 cards.
    vec![
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 5).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 15).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 20).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 2),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 4).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 12).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 28).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 15).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 5).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 8).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 24).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 10).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 3, 1).unwrap(), 3),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 1),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 1),
            (NaiveDate::from_ymd_opt(2023, 1, 3).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 6).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 16).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 20).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 3).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 8).unwrap(), 2),
            (NaiveDate::from_ymd_opt(2023, 1, 10).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 22).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 5).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 25).unwrap(), 3),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 9).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 19).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 5).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 25).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 2),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 5).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 15).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 30).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 15).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 5).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 4).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 14).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 20).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 1),
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 1),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 3).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 7).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 15).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 31).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 15).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 3, 5).unwrap(), 3),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 10).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 20).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 5).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 25).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 15).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 1),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 2),
            (NaiveDate::from_ymd_opt(2023, 1, 3).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 4).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 10).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 20).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 5).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 25).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 5).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 15).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 30).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 15).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 5).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 2),
            (NaiveDate::from_ymd_opt(2023, 1, 3).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 7).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 17).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 1).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 20).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 10).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 12).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 25).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 10).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 1).unwrap(), 4),
        ],
    ]
    .iter()
    .cloned()
    .cycle()
    .take(100)
    .collect()
}

fn convert_to_fsrs_item(history: Vec<(NaiveDate, u32)>) -> Vec<FSRSItem> {
    let mut reviews = Vec::new();
    let mut last_date = history[0].0;
    let mut items = Vec::new();

    for (date, rating) in history {
        let delta_t = (date - last_date).num_days() as u32;
        reviews.push(FSRSReview { rating, delta_t });
        items.push(FSRSItem {
            reviews: reviews.clone(),
        });
        last_date = date;
    }

    items
        .into_iter()
        .filter(|item| item.long_term_review_cnt() > 0)
        .collect()
}
