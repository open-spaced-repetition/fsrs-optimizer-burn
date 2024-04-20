use crate::convertor_tests::RevlogReviewKind::*;
use crate::dataset::FSRSBatcher;
use crate::dataset::{FSRSItem, FSRSReview};
use crate::test_helpers::NdArrayAutodiff;
use burn::backend::ndarray::NdArrayDevice;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::data::dataset::InMemDataset;
use burn::tensor::Data;
use chrono::prelude::*;
use chrono_tz::Tz;
use itertools::Itertools;
use rusqlite::Connection;
use rusqlite::{Result, Row};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RevlogEntry {
    pub id: i64,
    pub cid: i64,
    pub usn: i32,
    /// - In the V1 scheduler, 3 represents easy in the learning case.
    /// - 0 represents manual rescheduling.
    pub button_chosen: u8,
    /// Positive values are in days, negative values in seconds.
    pub interval: i32,
    /// Positive values are in days, negative values in seconds.
    pub last_interval: i32,
    /// Card's ease after answering, stored as 10x the %, eg 2500 represents
    /// 250%.
    pub ease_factor: u32,
    /// Amount of milliseconds taken to answer the card.
    pub taken_millis: u32,
    pub review_kind: RevlogReviewKind,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum RevlogReviewKind {
    #[default]
    Learning = 0,
    Review = 1,
    Relearning = 2,
    /// Old Anki versions called this "Cram" or "Early", and assigned it when
    /// reviewing cards ahead. It is now only used for filtered decks with
    /// rescheduling disabled.
    Filtered = 3,
    Manual = 4,
}

impl rusqlite::types::FromSql for RevlogReviewKind {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
        let rusqlite::types::ValueRef::Integer(i) = value else {
            return Err(rusqlite::types::FromSqlError::InvalidType);
        };
        match i {
            0 => Ok(RevlogReviewKind::Learning),
            1 => Ok(RevlogReviewKind::Review),
            2 => Ok(RevlogReviewKind::Relearning),
            3 => Ok(RevlogReviewKind::Filtered),
            4 => Ok(RevlogReviewKind::Manual),
            _ => Err(rusqlite::types::FromSqlError::InvalidType),
        }
    }
}

impl TryFrom<&Row<'_>> for RevlogEntry {
    type Error = rusqlite::Error;
    fn try_from(row: &Row<'_>) -> Result<Self> {
        Ok(RevlogEntry {
            id: row.get(0)?,
            cid: row.get(1)?,
            usn: row.get(2)?,
            button_chosen: row.get(3)?,
            interval: row.get(4)?,
            last_interval: row.get(5)?,
            ease_factor: row.get(6)?,
            taken_millis: row.get(7)?,
            review_kind: row.get(8)?,
        })
    }
}

fn filter_out_cram(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
    entries
        .into_iter()
        .filter(|entry| entry.review_kind != Filtered || entry.ease_factor != 0)
        .collect()
}

fn filter_out_manual(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
    entries
        .into_iter()
        .filter(|entry| entry.review_kind != Manual && entry.button_chosen != 0)
        .collect()
}

fn remove_revlog_before_last_first_learn(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
    let mut last_first_learn_index = 0;
    for (index, entry) in entries.iter().enumerate().rev() {
        if entry.review_kind == Learning {
            last_first_learn_index = index;
        } else if last_first_learn_index != 0 {
            break;
        }
    }
    if entries[last_first_learn_index].review_kind == Learning {
        entries[last_first_learn_index..].to_vec()
    } else {
        vec![]
    }
}

fn convert_to_date(timestamp: i64, next_day_starts_at: i64, timezone: Tz) -> NaiveDate {
    let timestamp_seconds = timestamp - next_day_starts_at * 3600 * 1000;
    let datetime = Utc
        .timestamp_millis_opt(timestamp_seconds)
        .unwrap()
        .with_timezone(&timezone);
    datetime.date_naive()
}

fn keep_first_revlog_same_date(
    mut entries: Vec<RevlogEntry>,
    next_day_starts_at: i64,
    timezone: Tz,
) -> Vec<RevlogEntry> {
    let mut unique_dates = std::collections::HashSet::new();
    entries.retain(|entry| {
        let date = convert_to_date(entry.id, next_day_starts_at, timezone);
        unique_dates.insert(date)
    });
    entries
}

/// Given a list of revlog entries for a single card with length n, we create
/// n-1 FSRS items, where each item contains the history of the preceding reviews.

fn convert_to_fsrs_items(
    mut entries: Vec<RevlogEntry>,
    next_day_starts_at: i64,
    timezone: Tz,
) -> Option<Vec<FSRSItem>> {
    // entries = filter_out_cram(entries);
    // entries = filter_out_manual(entries);
    entries = remove_revlog_before_last_first_learn(entries);
    entries = keep_first_revlog_same_date(entries, next_day_starts_at, timezone);

    for i in 1..entries.len() {
        let date_current = convert_to_date(entries[i].id, next_day_starts_at, timezone);
        let date_previous = convert_to_date(entries[i - 1].id, next_day_starts_at, timezone);
        entries[i].last_interval = (date_current - date_previous).num_days() as i32;
    }

    Some(
        entries
            .iter()
            .enumerate()
            .skip(1)
            .map(|(idx, _)| {
                let reviews = entries
                    .iter()
                    .take(idx + 1)
                    .map(|r| FSRSReview {
                        rating: r.button_chosen as u32,
                        delta_t: r.last_interval.max(0) as u32,
                    })
                    .collect();
                FSRSItem { reviews }
            })
            .collect(),
    )
}

/// Convert a series of revlog entries sorted by card id into FSRS items.
pub(crate) fn anki_to_fsrs(revlogs: Vec<RevlogEntry>) -> Vec<FSRSItem> {
    let mut revlogs = revlogs
        .into_iter()
        .group_by(|r| r.cid)
        .into_iter()
        .filter_map(|(_cid, entries)| {
            convert_to_fsrs_items(entries.collect(), 4, Tz::Asia__Shanghai)
        })
        .flatten()
        .collect_vec();
    revlogs.sort_by_cached_key(|r| r.reviews.len());
    revlogs
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RevlogCsv {
    // card_id,review_time,review_rating,review_state,review_duration
    pub card_id: i64,
    pub review_time: i64,
    pub review_rating: u32,
    pub review_state: u32,
    pub review_duration: u32,
}

pub(crate) fn data_from_csv() -> Vec<FSRSItem> {
    const CSV_FILE: &str = "tests/data/revlog.csv";
    let rdr = csv::ReaderBuilder::new();
    let dataset = InMemDataset::<RevlogCsv>::from_csv(CSV_FILE, &rdr).unwrap();
    let revlogs: Vec<RevlogEntry> = dataset
        .iter()
        .map(|r| RevlogEntry {
            id: r.review_time,
            cid: r.card_id,
            button_chosen: r.review_rating as u8,
            taken_millis: r.review_duration,
            review_kind: match r.review_state {
                0 => RevlogReviewKind::Learning,
                1 => RevlogReviewKind::Learning,
                2 => RevlogReviewKind::Review,
                3 => RevlogReviewKind::Relearning,
                4 => RevlogReviewKind::Filtered,
                5 => RevlogReviewKind::Manual,
                _ => panic!("Invalid review state"),
            },
            ..Default::default()
        })
        .collect();
    dbg!(revlogs.len());
    let fsrs_items = anki_to_fsrs(revlogs);
    dbg!(fsrs_items.len());
    fsrs_items
}

pub(crate) fn anki21_sample_file_converted_to_fsrs() -> Vec<FSRSItem> {
    anki_to_fsrs(read_collection().expect("read error"))
}

fn read_collection() -> Result<Vec<RevlogEntry>> {
    let db = Connection::open("tests/data/collection.anki21")?;
    let filter_out_suspended_cards = false;
    let filter_out_flags = [];
    let flags_str = if !filter_out_flags.is_empty() {
        format!(
            "AND flags NOT IN ({})",
            filter_out_flags
                .iter()
                .map(|x: &i32| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        "".to_string()
    };

    let suspended_cards_str = if filter_out_suspended_cards {
        "AND queue != -1"
    } else {
        ""
    };

    let current_timestamp = Utc::now().timestamp() * 1000;
    // This sql query will be remove in the futrue. See https://github.com/open-spaced-repetition/fsrs-optimizer-burn/pull/14#issuecomment-1685895643
    let revlogs = db
        .prepare_cached(&format!(
            "SELECT *
        FROM revlog
        WHERE id < ?1
        AND cid < ?2
        AND cid IN (
            SELECT id
            FROM cards
            WHERE queue != 0
            {suspended_cards_str}
            {flags_str}
        )
        AND ease BETWEEN 1 AND 4
        AND (
            type != 3
            OR factor != 0
        )
        order by cid"
        ))?
        .query_and_then((current_timestamp, current_timestamp), |x| x.try_into())?
        .collect::<Result<Vec<_>>>()?;
    Ok(revlogs)
}

#[test]
fn extract_simulator_config_from_revlog() {
    let revlogs = read_collection().unwrap();
    let first_rating_count = revlogs
        .iter()
        .filter(|r| {
            r.review_kind == RevlogReviewKind::Learning
                && r.ease_factor == 0
                && r.button_chosen >= 1
        })
        .counts_by(|r| r.button_chosen);
    let total_ratings = first_rating_count.values().sum::<usize>() as f32;
    let first_rating_prob = {
        let mut arr = [0.0; 4];
        first_rating_count
            .iter()
            .for_each(|(button_chosen, count)| {
                arr[*button_chosen as usize - 1] = *count as f32 / total_ratings
            });
        arr
    };
    assert_eq!(first_rating_prob, [0.15339181, 0.0, 0.15339181, 0.6932164,]);
    let review_rating_count = revlogs
        .iter()
        .filter(|r| r.review_kind == RevlogReviewKind::Review && r.button_chosen != 1)
        .counts_by(|r| r.button_chosen);
    let review_rating_prob = {
        let mut arr = [0.0; 3];
        review_rating_count
            .iter()
            .filter(|(&button_chosen, ..)| button_chosen >= 2)
            .for_each(|(button_chosen, count)| {
                arr[*button_chosen as usize - 2] =
                    *count as f32 / review_rating_count.values().sum::<usize>() as f32;
            });
        arr
    };
    assert_eq!(review_rating_prob, [0.07380187, 0.90085745, 0.025340684,]);

    let recall_costs = {
        let mut arr = [0.0; 5];
        revlogs
            .iter()
            .filter(|r| r.review_kind == RevlogReviewKind::Review && r.button_chosen > 0)
            .sorted_by(|a, b| a.button_chosen.cmp(&b.button_chosen))
            .group_by(|r| r.button_chosen)
            .into_iter()
            .for_each(|(button_chosen, group)| {
                let group_vec = group.into_iter().map(|r| r.taken_millis).collect_vec();
                let average_secs =
                    group_vec.iter().sum::<u32>() as f32 / group_vec.len() as f32 / 1000.0;
                arr[button_chosen as usize - 1] = average_secs
            });
        arr
    };
    let learn_cost = {
        let revlogs_filter = revlogs
            .iter()
            .filter(|r| r.review_kind == RevlogReviewKind::Learning && r.ease_factor == 0)
            .map(|r| r.taken_millis);
        revlogs_filter.clone().sum::<u32>() as f32 / revlogs_filter.count() as f32 / 1000.0
    };
    assert_eq!(learn_cost, 8.980446);

    let forget_cost = {
        let review_kind_to_total_millis = revlogs
            .iter()
            .sorted_by(|a, b| a.cid.cmp(&b.cid).then(a.id.cmp(&b.id)))
            .group_by(|r| r.review_kind)
            /*
                for example:
                o  x x  o o x x x o o x x o x
                  |<->|    |<--->|   |<->| |<>|
                x means forgotten, there are 4 consecutive sets of internal relearning in this card.
                So each group is counted separately, and each group is summed up internally.(following code)
                Finally averaging all groups, so sort by cid and id.
            */
            .into_iter()
            .map(|(review_kind, group)| {
                let total_millis: u32 = group.into_iter().map(|r| r.taken_millis).sum();
                (review_kind, total_millis)
            })
            .collect_vec();
        let mut group_sec_by_review_kind: [Vec<_>; 5] = Default::default();
        for (review_kind, sec) in review_kind_to_total_millis.into_iter() {
            group_sec_by_review_kind[review_kind as usize].push(sec)
        }
        let mut arr = [0.0; 5];
        group_sec_by_review_kind
            .iter()
            .enumerate()
            .for_each(|(review_kind, group)| {
                let average_secs = group.iter().sum::<u32>() as f32 / group.len() as f32 / 1000.0;
                arr[review_kind] = average_secs
            });
        arr
    };

    let forget_cost = forget_cost[RevlogReviewKind::Relearning as usize] + recall_costs[0];
    assert_eq!(forget_cost, 23.481838);

    assert_eq!(recall_costs[1..=3], [9.047336, 7.774851, 5.149275,]);
}

// This test currently expects the following .anki21 file to be placed in tests/data/:
// https://github.com/open-spaced-repetition/fsrs-optimizer-burn/files/12394182/collection.anki21.zip
#[test]
fn conversion_works() {
    let revlogs = read_collection().unwrap();
    let single_card_revlog = vec![revlogs
        .iter()
        .filter(|r| r.cid == 1528947214762)
        .cloned()
        .collect_vec()];
    assert_eq!(revlogs.len(), 24394);
    let fsrs_items = anki_to_fsrs(revlogs);
    assert_eq!(fsrs_items.len(), 14290);
    assert_eq!(
        fsrs_items.iter().map(|x| x.reviews.len()).sum::<usize>(),
        49382 + 14290
    );

    // convert a subset and check it matches expectations
    let mut fsrs_items = single_card_revlog
        .into_iter()
        .filter_map(|entries| convert_to_fsrs_items(entries, 4, Tz::Asia__Shanghai))
        .flatten()
        .collect_vec();
    assert_eq!(
        fsrs_items,
        [
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 5
                    }
                ],
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 5
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 10
                    }
                ],
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 5
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 10
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 22
                    }
                ],
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 5
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 10
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 22
                    },
                    FSRSReview {
                        rating: 2,
                        delta_t: 56
                    }
                ],
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 5
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 10
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 22
                    },
                    FSRSReview {
                        rating: 2,
                        delta_t: 56
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 64
                    }
                ],
            }
        ]
    );

    let device = NdArrayDevice::Cpu;
    let batcher = FSRSBatcher::<NdArrayAutodiff>::new(device);
    let res = batcher.batch(vec![fsrs_items.pop().unwrap()]);
    assert_eq!(res.delta_ts.into_scalar(), 64.0);
    assert_eq!(
        res.r_historys.squeeze(1).to_data(),
        Data::from([3.0, 3.0, 3.0, 3.0, 2.0])
    );
    assert_eq!(
        res.t_historys.squeeze(1).to_data(),
        Data::from([0.0, 5.0, 10.0, 22.0, 56.0])
    );
    assert_eq!(res.labels.to_data(), Data::from([1]));
}

#[test]
fn ordering_of_inputs_should_not_change() {
    let revlogs = anki21_sample_file_converted_to_fsrs();
    assert_eq!(
        revlogs[0],
        FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 4,
                    delta_t: 0
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 3
                }
            ]
        }
    );
}

const NEXT_DAY_AT: i64 = 86400 * 100;

fn revlog(review_kind: RevlogReviewKind, days_ago: i64) -> RevlogEntry {
    RevlogEntry {
        review_kind,
        id: ((NEXT_DAY_AT - days_ago * 86400) * 1000),
        button_chosen: 3,
        ..Default::default()
    }
}

#[test]
fn delta_t_is_correct() -> Result<()> {
    assert_eq!(
        convert_to_fsrs_items(
            vec![
                revlog(RevlogReviewKind::Learning, 1),
                revlog(RevlogReviewKind::Review, 0)
            ],
            NEXT_DAY_AT,
            Tz::Asia__Shanghai
        ),
        Some(vec![FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 3,
                    delta_t: 0
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 1
                }
            ]
        }])
    );

    assert_eq!(
        convert_to_fsrs_items(
            vec![
                revlog(RevlogReviewKind::Learning, 15),
                revlog(RevlogReviewKind::Learning, 13),
                revlog(RevlogReviewKind::Review, 10),
                revlog(RevlogReviewKind::Review, 5)
            ],
            NEXT_DAY_AT,
            Tz::Asia__Shanghai
        ),
        Some(vec![
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 2
                    }
                ]
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 2
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 3
                    }
                ]
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 2
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 3
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 5
                    }
                ]
            }
        ])
    );

    Ok(())
}
#[test]
fn test_filter_out_cram() {
    let revlog_vec = vec![
        RevlogEntry {
            id: 1581372672843,
            cid: 1559078645460,
            usn: 5212,
            button_chosen: 1,
            interval: -60,
            last_interval: -60,
            ease_factor: 0,
            taken_millis: 38942,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1694423345134,
            cid: 1559078645460,
            usn: 12394,
            button_chosen: 3,
            interval: 131,
            last_interval: 73,
            ease_factor: 1750,
            taken_millis: 30443,
            review_kind: Review,
        },
        RevlogEntry {
            id: 1694422345134,
            cid: 1559078645460,
            usn: -1,
            button_chosen: 3,
            interval: -1200,
            last_interval: -600,
            ease_factor: 0,
            taken_millis: 2231,
            review_kind: Filtered,
        },
        RevlogEntry {
            id: 1694423345134,
            cid: 1559078645460,
            usn: 12394,
            button_chosen: 3,
            interval: 131,
            last_interval: 73,
            ease_factor: 1750,
            taken_millis: 30443,
            review_kind: Review,
        },
    ];
    let revlog_vec = filter_out_cram(revlog_vec);
    assert_eq!(
        revlog_vec,
        vec![
            RevlogEntry {
                id: 1581372672843,
                cid: 1559078645460,
                usn: 5212,
                button_chosen: 1,
                interval: -60,
                last_interval: -60,
                ease_factor: 0,
                taken_millis: 38942,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1694423345134,
                cid: 1559078645460,
                usn: 12394,
                button_chosen: 3,
                interval: 131,
                last_interval: 73,
                ease_factor: 1750,
                taken_millis: 30443,
                review_kind: Review,
            },
            RevlogEntry {
                id: 1694423345134,
                cid: 1559078645460,
                usn: 12394,
                button_chosen: 3,
                interval: 131,
                last_interval: 73,
                ease_factor: 1750,
                taken_millis: 30443,
                review_kind: Review,
            },
        ]
    );
}

#[test]
fn test_filter_out_manual() {
    let revlog_vec = vec![
        RevlogEntry {
            id: 1581544939407,
            cid: 1559457421654,
            usn: 5226,
            button_chosen: 3,
            interval: -600,
            last_interval: -60,
            ease_factor: 0,
            taken_millis: 6979,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581545802767,
            cid: 1559457421654,
            usn: 5226,
            button_chosen: 3,
            interval: 1,
            last_interval: -600,
            ease_factor: 2500,
            taken_millis: 21297,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581629454703,
            cid: 1559457421654,
            usn: -1,
            button_chosen: 0,
            interval: 302,
            last_interval: 302,
            ease_factor: 2150,
            taken_millis: 0,
            review_kind: Manual,
        },
        RevlogEntry {
            id: 1582012577455,
            cid: 1559457421654,
            usn: 11054,
            button_chosen: 3,
            interval: 270,
            last_interval: 161,
            ease_factor: 2150,
            taken_millis: 31190,
            review_kind: Review,
        },
    ];
    let revlog_vec = filter_out_manual(revlog_vec);
    assert_eq!(
        revlog_vec,
        vec![
            RevlogEntry {
                id: 1581544939407,
                cid: 1559457421654,
                usn: 5226,
                button_chosen: 3,
                interval: -600,
                last_interval: -60,
                ease_factor: 0,
                taken_millis: 6979,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1581545802767,
                cid: 1559457421654,
                usn: 5226,
                button_chosen: 3,
                interval: 1,
                last_interval: -600,
                ease_factor: 2500,
                taken_millis: 21297,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1582012577455,
                cid: 1559457421654,
                usn: 11054,
                button_chosen: 3,
                interval: 270,
                last_interval: 161,
                ease_factor: 2150,
                taken_millis: 31190,
                review_kind: Review,
            },
        ]
    );
}

#[test]
fn test_remove_revlog_before_last_first_learn() {
    let revlog_vec = vec![
        RevlogEntry {
            id: 1250450107406,
            cid: 1249409627511,
            usn: 0,
            button_chosen: 3,
            interval: 44,
            last_interval: 14,
            ease_factor: 2000,
            taken_millis: 9827,
            review_kind: Review,
        },
        RevlogEntry {
            id: 1254407572859,
            cid: 1249409627511,
            usn: 0,
            button_chosen: 3,
            interval: 90,
            last_interval: 44,
            ease_factor: 2000,
            taken_millis: 11187,
            review_kind: Review,
        },
        RevlogEntry {
            id: 1601346317001,
            cid: 1249409627511,
            usn: 6235,
            button_chosen: 2,
            interval: -330,
            last_interval: -60,
            ease_factor: 2500,
            taken_millis: 36376,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1601346783928,
            cid: 1249409627511,
            usn: 6235,
            button_chosen: 3,
            interval: -600,
            last_interval: -60,
            ease_factor: 2500,
            taken_millis: 16249,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1601349355546,
            cid: 1249409627511,
            usn: 6235,
            button_chosen: 3,
            interval: 1,
            last_interval: -600,
            ease_factor: 2500,
            taken_millis: 13272,
            review_kind: Learning,
        },
    ];
    let revlog_vec = remove_revlog_before_last_first_learn(revlog_vec);
    // dbg!(&revlog_vec);
    assert_eq!(
        revlog_vec,
        vec![
            RevlogEntry {
                id: 1601346317001,
                cid: 1249409627511,
                usn: 6235,
                button_chosen: 2,
                interval: -330,
                last_interval: -60,
                ease_factor: 2500,
                taken_millis: 36376,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1601346783928,
                cid: 1249409627511,
                usn: 6235,
                button_chosen: 3,
                interval: -600,
                last_interval: -60,
                ease_factor: 2500,
                taken_millis: 16249,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1601349355546,
                cid: 1249409627511,
                usn: 6235,
                button_chosen: 3,
                interval: 1,
                last_interval: -600,
                ease_factor: 2500,
                taken_millis: 13272,
                review_kind: Learning,
            },
        ]
    );

    let revlog_vec = vec![
        RevlogEntry {
            id: 1224652906547,
            cid: 1224124699235,
            usn: 0,
            button_chosen: 3,
            interval: 0,
            last_interval: 0,
            ease_factor: 2498,
            taken_millis: 50562,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1225062601238,
            cid: 1224124699235,
            usn: 0,
            button_chosen: 4,
            interval: 4,
            last_interval: 0,
            ease_factor: 2498,
            taken_millis: 22562,
            review_kind: Relearning,
        },
        RevlogEntry {
            id: 1389281322102,
            cid: 1224124699235,
            usn: 866,
            button_chosen: 4,
            interval: 4,
            last_interval: -60,
            ease_factor: 2500,
            taken_millis: 22559,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1389894643706,
            cid: 1224124699235,
            usn: 889,
            button_chosen: 3,
            interval: 9,
            last_interval: 4,
            ease_factor: 2500,
            taken_millis: 14763,
            review_kind: Review,
        },
    ];

    let revlog_vec = remove_revlog_before_last_first_learn(revlog_vec);
    assert_eq!(
        revlog_vec,
        vec![
            RevlogEntry {
                id: 1389281322102,
                cid: 1224124699235,
                usn: 866,
                button_chosen: 4,
                interval: 4,
                last_interval: -60,
                ease_factor: 2500,
                taken_millis: 22559,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1389894643706,
                cid: 1224124699235,
                usn: 889,
                button_chosen: 3,
                interval: 9,
                last_interval: 4,
                ease_factor: 2500,
                taken_millis: 14763,
                review_kind: Review,
            },
        ]
    );
}

#[test]
fn test_keep_first_revlog_same_date() {
    let revlog_vec = vec![
        RevlogEntry {
            id: 1581372095493,
            cid: 1559076329057,
            usn: 5212,
            button_chosen: 1,
            interval: -60,
            last_interval: -60,
            ease_factor: 0,
            taken_millis: 60000,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581372260598,
            cid: 1559076329057,
            usn: 5212,
            button_chosen: 3,
            interval: -600,
            last_interval: -60,
            ease_factor: 0,
            taken_millis: 46425,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581406251414,
            cid: 1559076329057,
            usn: 5213,
            button_chosen: 2,
            interval: -600,
            last_interval: -600,
            ease_factor: 0,
            taken_millis: 17110,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581407568344,
            cid: 1559076329057,
            usn: 5213,
            button_chosen: 3,
            interval: 1,
            last_interval: -600,
            ease_factor: 2500,
            taken_millis: 8861,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581454426227,
            cid: 1559076329057,
            usn: 5215,
            button_chosen: 3,
            interval: 3,
            last_interval: 1,
            ease_factor: 2500,
            taken_millis: 13128,
            review_kind: Review,
        },
    ];
    let revlog_vec = keep_first_revlog_same_date(revlog_vec, 4, Tz::Asia__Shanghai);
    assert_eq!(
        revlog_vec,
        vec![
            RevlogEntry {
                id: 1581372095493,
                cid: 1559076329057,
                usn: 5212,
                button_chosen: 1,
                interval: -60,
                last_interval: -60,
                ease_factor: 0,
                taken_millis: 60000,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1581454426227,
                cid: 1559076329057,
                usn: 5215,
                button_chosen: 3,
                interval: 3,
                last_interval: 1,
                ease_factor: 2500,
                taken_millis: 13128,
                review_kind: Review,
            },
        ]
    )
}
