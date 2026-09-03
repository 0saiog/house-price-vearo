//! Parsers for the messy text columns.
//!
//! All of these come from actually looking at the file rather than the column
//! descriptions. Prices are things like `"42 Lac"` and `"Call for Price"`, areas
//! show up in ten different units, `Floor` is `"3 out of 10"` with some
//! `"Ground"` and basements thrown in, and `Car Parking` sometimes has a
//! trailing comma.

/// One lakh in rupees.
const LAC: f64 = 100_000.0;
/// One crore in rupees.
const CR: f64 = 10_000_000.0;

/// Square feet per unit, for every area unit that appears in the file.
const AREA_UNITS: &[(&str, f64)] = &[
    ("sqft", 1.0),
    ("sqyrd", 9.0),
    ("sqm", 10.763_910_4),
    ("acre", 43_560.0),
    ("marla", 272.25),
    ("kanal", 5_445.0),
    ("ground", 2_400.0),
    ("cent", 435.6),
    ("bigha", 27_000.0),
    ("biswa1", 1_350.0),
    ("biswa2", 1_350.0),
    ("aankadam", 72.0),
    ("guntha", 1_089.0),
    ("hectare", 107_639.104),
    ("rood", 10_890.0),
    ("chatak", 180.0),
    ("perch", 272.25),
    ("are", 1_076.391_04),
];

/// Parses `"42 Lac"`, `"1.40 Cr"` or `"3,50,000"` into rupees.
///
/// Returns `None` for `"Call for Price"`, blanks and anything else unparseable -
/// a listing with no usable price cannot be a training row.
#[must_use]
pub fn parse_amount(raw: &str) -> Option<f64> {
    let s = raw.trim().to_lowercase();
    if s.is_empty() || s.contains("call") {
        return None;
    }
    let (multiplier, number) = if let Some(rest) = s.strip_suffix("lac") {
        (LAC, rest)
    } else if let Some(rest) = s.strip_suffix("cr") {
        (CR, rest)
    } else {
        (1.0, s.as_str())
    };
    let n: f64 = number.replace(',', "").trim().parse().ok()?;
    (n.is_finite() && n > 0.0).then_some(n * multiplier)
}

/// Parses `"1200 sqft"`, `"140 sqyrd"`, `"90 sqm"` into square feet.
///
/// Unknown units return `None` instead of a wrong number.
#[must_use]
pub fn parse_area_sqft(raw: &str) -> Option<f64> {
    let s = raw.trim().to_lowercase();
    if s.is_empty() {
        return None;
    }
    let mut parts = s.split_whitespace();
    let n: f64 = parts.next()?.replace(',', "").parse().ok()?;
    let unit = parts.next().unwrap_or("sqft");
    let factor = AREA_UNITS
        .iter()
        .find_map(|&(u, f)| (u == unit).then_some(f))?;
    let sqft = n * factor;
    (sqft.is_finite() && sqft > 0.0).then_some(sqft)
}

/// Parsed `Floor`: the storey the flat is on, and how many the building has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Floor {
    /// Storey number. Ground is 0, basements are negative.
    pub number: Option<i32>,
    /// Storeys in the building.
    pub total: Option<i32>,
}

/// Parses `"3 out of 10"`, `"Ground out of 4"`, `"Basement"`, `"2"`.
#[must_use]
pub fn parse_floor(raw: &str) -> Floor {
    let s = raw.trim().to_lowercase();
    if s.is_empty() {
        return Floor {
            number: None,
            total: None,
        };
    }
    let (head, tail) = match s.split_once(" out of ") {
        Some((h, t)) => (h.trim(), Some(t.trim())),
        None => (s.as_str(), None),
    };
    let number = match head {
        "ground" => Some(0),
        "lower basement" => Some(-2),
        "basement" | "upper basement" => Some(-1),
        other => other.parse().ok(),
    };
    Floor {
        number,
        total: tail.and_then(|t| t.parse().ok()),
    }
}

/// Parses a small count column such as `Bathroom` or `Balcony`.
///
/// `"> 10"` becomes 11, which keeps the ordering meaningful without inventing a
/// precise value the file does not have.
#[must_use]
pub fn parse_count(raw: &str) -> Option<f64> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(rest) = s.strip_prefix('>') {
        return rest.trim().parse::<f64>().ok().map(|n| n + 1.0);
    }
    s.parse().ok()
}

/// Parsed `Car Parking`: how many spaces, and whether any is covered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parking {
    /// Number of spaces.
    pub count: Option<f64>,
    /// Whether the listing says "Covered".
    pub covered: bool,
}

/// Parses `"1 Covered"`, `"2 Open"`, `"1 Covered,"`.
#[must_use]
pub fn parse_parking(raw: &str) -> Parking {
    let s = raw.trim().trim_end_matches(',').trim().to_lowercase();
    if s.is_empty() {
        return Parking {
            count: None,
            covered: false,
        };
    }
    let mut parts = s.split_whitespace();
    let count = parts.next().and_then(|n| n.parse::<f64>().ok());
    Parking {
        count,
        covered: s.contains("covered"),
    }
}

/// The three things `overlooking` can mention, as independent flags.
///
/// The column is a comma-separated set in an inconsistent order
/// (`"Garden/Park, Pool, Main Road"`, `"Pool, Main Road"`), so one-hot encoding
/// the raw string would make 20 categories out of 3 facts.
#[must_use]
pub fn parse_overlooking(raw: &str) -> [f64; 3] {
    let s = raw.to_lowercase();
    [
        f64::from(u8::from(s.contains("garden"))),
        f64::from(u8::from(s.contains("pool"))),
        f64::from(u8::from(s.contains("main road"))),
    ]
}

/// Normalises `facing`, which spells the same direction two ways
/// (`"South - West"` and `"South -West"`).
#[must_use]
pub fn normalize_facing(raw: &str) -> String {
    let s = raw.trim().to_lowercase().replace(['-', ' '], "");
    if s.is_empty() {
        "missing".to_string()
    } else {
        s
    }
}

/// Normalises any other categorical value, mapping blanks to `"missing"` so the
/// absence itself becomes a category instead of a dropped row.
#[must_use]
pub fn normalize_category(raw: &str) -> String {
    let s = raw.trim().to_lowercase();
    if s.is_empty() {
        "missing".to_string()
    } else {
        s
    }
}

/// Normalises a free-text name into a stable categorical key.
///
/// Localities and society names contain commas, hyphens and inconsistent
/// whitespace. Keeping only lowercase alphanumeric tokens makes a value typed
/// in the API match the same value extracted from a training title.
#[must_use]
pub fn normalize_text_key(raw: &str) -> String {
    let cleaned: String = raw
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    let key = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    if key.is_empty() {
        "missing".to_string()
    } else {
        key
    }
}

/// Extracts bedroom count from titles such as `"3 BHK ..."` and studios.
#[must_use]
pub fn parse_bedrooms(title: &str) -> Option<f64> {
    let key = normalize_text_key(title);
    let tokens: Vec<&str> = key.split_whitespace().collect();
    for pair in tokens.windows(2) {
        if pair[1] == "bhk" || pair[1] == "rk" {
            return pair[0].parse::<f64>().ok().filter(|n| *n > 0.0);
        }
    }
    key.contains("studio").then_some(1.0)
}

/// Coarse property type extracted from the title.
///
/// Specific forms are checked before generic `house`, `flat` and `apartment`
/// matches so `studio apartment` and `farm house` remain distinct.
#[must_use]
pub fn property_type(title: &str) -> String {
    let key = normalize_text_key(title);
    let kind = if key.contains("studio") {
        "studio"
    } else if key.contains("penthouse") {
        "penthouse"
    } else if key.contains("builder floor") {
        "builder floor"
    } else if key.contains("farm house") || key.contains("farmhouse") {
        "farm house"
    } else if key.contains("villa") {
        "villa"
    } else if key.contains("plot") || key.contains("land") {
        "plot"
    } else if key.contains("house") {
        "house"
    } else if key.contains("flat") {
        "flat"
    } else if key.contains("apartment") {
        "apartment"
    } else {
        "other"
    };
    kind.to_string()
}

/// Extracts a locality from the part of a title after `for sale`.
///
/// Titles commonly end in `"for sale in <society> <locality>, <city>"`.
/// Society and city are already separate inputs, so stripping matching prefix
/// and suffix text leaves the neighbourhood signal the old model was missing.
#[must_use]
pub fn extract_locality(title: &str, society: &str, city: &str) -> String {
    let title = normalize_text_key(title);
    let Some(pos) = title.find("for sale") else {
        return "missing".to_string();
    };
    let mut locality = title[pos + "for sale".len()..].trim();
    locality = locality.strip_prefix("in ").unwrap_or(locality).trim();

    let society = normalize_text_key(society);
    if society != "missing" {
        locality = locality.strip_prefix(&society).unwrap_or(locality).trim();
    }

    let city = normalize_text_key(city);
    if city != "missing" {
        if locality == city {
            return "missing".to_string();
        }
        locality = locality
            .strip_suffix(&format!(" {city}"))
            .unwrap_or(locality)
            .trim();
    }

    normalize_text_key(locality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amounts() {
        assert_eq!(parse_amount("42 Lac "), Some(4_200_000.0));
        assert_eq!(parse_amount("1.40 Cr"), Some(14_000_000.0));
        assert_eq!(parse_amount("3,50,000"), Some(350_000.0));
        assert_eq!(parse_amount("Call for Price"), None);
        assert_eq!(parse_amount(""), None);
        assert_eq!(parse_amount("0 Lac"), None);
    }

    #[test]
    fn areas() {
        assert_eq!(parse_area_sqft("1200 sqft"), Some(1200.0));
        assert_eq!(parse_area_sqft("140 sqyrd"), Some(1260.0));
        assert!((parse_area_sqft("90 sqm").unwrap() - 968.75).abs() < 0.1);
        assert_eq!(
            parse_area_sqft("2 furlong"),
            None,
            "unknown units must not guess"
        );
        assert_eq!(parse_area_sqft(""), None);
    }

    #[test]
    fn floors() {
        assert_eq!(
            parse_floor("3 out of 10"),
            Floor {
                number: Some(3),
                total: Some(10)
            }
        );
        assert_eq!(
            parse_floor("Ground out of 4"),
            Floor {
                number: Some(0),
                total: Some(4)
            }
        );
        assert_eq!(
            parse_floor("Upper Basement out of 3"),
            Floor {
                number: Some(-1),
                total: Some(3)
            }
        );
        assert_eq!(
            parse_floor("2"),
            Floor {
                number: Some(2),
                total: None
            }
        );
        assert_eq!(
            parse_floor(""),
            Floor {
                number: None,
                total: None
            }
        );
    }

    #[test]
    fn counts_and_parking() {
        assert_eq!(parse_count("3"), Some(3.0));
        assert_eq!(parse_count("> 10"), Some(11.0));
        assert_eq!(parse_count(""), None);
        assert_eq!(
            parse_parking("1 Covered,"),
            Parking {
                count: Some(1.0),
                covered: true
            }
        );
        assert_eq!(
            parse_parking("2 Open"),
            Parking {
                count: Some(2.0),
                covered: false
            }
        );
        assert_eq!(
            parse_parking(""),
            Parking {
                count: None,
                covered: false
            }
        );
    }

    #[test]
    fn multi_label_and_categories() {
        assert_eq!(parse_overlooking("Garden/Park, Main Road"), [1.0, 0.0, 1.0]);
        assert_eq!(parse_overlooking("Pool"), [0.0, 1.0, 0.0]);
        assert_eq!(
            normalize_facing("South -West"),
            normalize_facing("South - West")
        );
        assert_eq!(normalize_facing(""), "missing");
        assert_eq!(normalize_category(" Semi-Furnished "), "semi-furnished");
    }

    #[test]
    fn title_features() {
        assert_eq!(parse_bedrooms("3 BHK Ready to Occupy Flat"), Some(3.0));
        assert_eq!(parse_bedrooms("Studio Apartment for sale"), Some(1.0));
        assert_eq!(property_type("Studio Apartment for sale"), "studio");
        assert_eq!(property_type("4 BHK Residential House"), "house");
        assert_eq!(
            extract_locality(
                "2 BHK Flat for sale in Santi Niwas Phase 6 Purbalok, Mukundapur, Kolkata",
                "Santi Niwas Phase 6",
                "kolkata"
            ),
            "purbalok mukundapur"
        );
        assert_eq!(
            extract_locality("2 BHK Flat for sale in Mumbai", "", "mumbai"),
            "missing"
        );
    }
}
