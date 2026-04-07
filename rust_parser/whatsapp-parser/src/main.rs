use clap::Parser;
use regex::Regex;
use serde::Serialize;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about = "Fast WhatsApp chat parser")]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long, default_value = "-")]
    output: String,  // "-" = stdout
}

#[derive(Serialize)]
struct ParsedMessage {
    message_start: Option<String>,  // ISO datetime or null
    sender: Option<String>,
    raw_text: String,
    cleaned_text: String,
    status: String,  // "NEW" | "IGNORED" | "DUPLICATE_LOCAL"
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    // Your exact HEADER_REGEX (ported 1:1)
    let header_re = Regex::new(
        r"(?x)
        ^[\u200E\u200F\u202A\u202B\u202C\u202D\u202E\s]*
        (?:\[)?
        (?P<date>\d{1,2}/\d{1,2}/\d{2,4})
        [,\s]+
        (?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM|am|pm))?)
        (?:\])?
        [\s\-–—:]*
        (?P<sender>[^:\n\r]{1,200}?)\s*:\s*
        "
    ).unwrap();

    let system_re = Regex::new(r"(?i)(end-to-end encrypted|message deleted|joined using|left the group|created group|changed the subject|<media omitted>)").unwrap();

    let input_file = File::open(&args.input)?;
    let reader = BufReader::new(input_file);

    let mut output: Box<dyn Write> = if args.output == "-" {
        Box::new(io::stdout())
    } else {
        Box::new(File::create(&args.output)?)
    };

    let mut lines = reader.lines();
    let text: String = lines.by_ref().map_while(Result::ok).collect::<Vec<_>>().join("\n");

    // Your exact splitting logic
    let matches: Vec<_> = header_re.find_iter(&text).collect();

    if matches.is_empty() {
        // fallback for files with no headers
        writeln!(output, "{}", serde_json::to_string(&ParsedMessage {
            message_start: None,
            sender: None,
            raw_text: text.clone(),
            cleaned_text: clean_block(&text, &system_re),
            status: "NEW".to_string(),
        })?)?;
        return Ok(());
    }

    for (i, m) in matches.iter().enumerate() {
        let start = m.start();
        let end = if i + 1 < matches.len() {
            matches[i + 1].start()
        } else {
            text.len()
        };

        let block = &text[start..end].trim();

        let caps = header_re.captures(block).unwrap();
        let date = caps.name("date").map(|m| m.as_str().to_string());
        let time = caps.name("time").map(|m| m.as_str().to_string());
        let sender = caps.name("sender").map(|m| m.as_str().trim().to_string());

        let cleaned = clean_block(block, &system_re);

        let status = if cleaned.is_empty() { "IGNORED" } else { "NEW" };

        let msg = ParsedMessage {
            message_start: date.and_then(|d| {
                // simple ISO conversion (you can expand with chrono if needed)
                Some(format!("{} {}", d, time.as_deref().unwrap_or("")))
            }),
            sender,
            raw_text: block.to_string(),
            cleaned_text: cleaned,
            status: status.to_string(),
        };

        writeln!(output, "{}", serde_json::to_string(&msg)?)?;
    }

    Ok(())
}

fn clean_block(block: &str, system_re: &Regex) -> String {
    let mut cleaned = block.to_string();

    // emoji stripping (Rust regex handles this very fast)
    let emoji_re = Regex::new(r"[\p{So}\p{Sk}\p{Cf}]").unwrap();
    cleaned = emoji_re.replace_all(&cleaned, " ").into_owned();

    // normalize whitespace (your exact logic)
    cleaned = cleaned.replace(['\u{00A0}', '\u{202F}'], " ")
                     .replace(['\u{200B}', '\u{200E}', '\u{200F}'], "");

    cleaned = regex::Regex::new(r"[ \t]+").unwrap().replace_all(&cleaned, " ").into_owned();

    // remove header part
    if let Some(m) = regex::Regex::new(r"^.*?:\s*").unwrap().find(&cleaned) {
        cleaned = cleaned[m.end()..].trim().to_string();
    }

    // remove system lines
    cleaned = cleaned.lines()
        .filter(|line| !system_re.is_match(line.trim()))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();

    cleaned
}