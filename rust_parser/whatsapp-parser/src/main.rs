use clap::Parser;
use regex::Regex;
use serde::Serialize;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use unicode_normalization::UnicodeNormalization;

#[derive(Parser)]
#[command(author, version, about = "Fast WhatsApp chat parser")]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long, default_value = "-")]
    output: String,
}

#[derive(Serialize)]
struct ParsedMessage {
    date_raw: Option<String>,
    time_raw: Option<String>,
    sender: Option<String>,
    raw_text: String,
    cleaned_text: String,
    status: String,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let header_re = Regex::new(
        r"(?mix)
        ^[\u200E\u200F\u202A\u202B\u202C\u202D\u202E\s]*
        (?:\[)?
        (?P<date>\d{1,2}/\d{1,2}/\d{2,4})
        [,\s]+
        (?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM|am|pm))?)
        (?:\])?
        [\s\-–—:]*
        (?P<sender>[^:\n\r]{1,200}?)\s*:\s*
        ",
    )
    .unwrap();

    let system_re = Regex::new(
        r"(?i)(end-to-end encrypted|message deleted|joined using|left the group|created group|changed the subject|<media omitted>)",
    )
    .unwrap();
    let emoji_re = Regex::new(r"\p{So}|\p{Sk}|\p{Cf}").unwrap();
    let space_re = Regex::new(r"[ \t]+").unwrap();

    let mut text = String::new();
    File::open(&args.input)?.read_to_string(&mut text)?;
    text = normalize_whitespace(&text, &space_re);

    let matches: Vec<_> = header_re.find_iter(&text).collect();

    let mut output: Box<dyn Write> = if args.output == "-" {
        Box::new(io::stdout())
    } else {
        Box::new(File::create(&args.output)?)
    };

    if matches.is_empty() {
        let cleaned = clean_block(&text, &header_re, &system_re, &emoji_re, &space_re);
        let status = if cleaned.is_empty() { "IGNORED" } else { "NEW" };
        let msg = ParsedMessage {
            date_raw: None,
            time_raw: None,
            sender: None,
            raw_text: text,
            cleaned_text: cleaned,
            status: status.to_string(),
        };
        writeln!(output, "{}", serde_json::to_string(&msg)?)?;
        return Ok(());
    }

    for (i, m) in matches.iter().enumerate() {
        let start = m.start();
        let end = if i + 1 < matches.len() {
            matches[i + 1].start()
        } else {
            text.len()
        };
        let block = text[start..end].trim();

        let (date_raw, time_raw, sender) = if let Some(caps) = header_re.captures(block) {
            (
                caps.name("date").map(|d| d.as_str().to_string()),
                caps.name("time").map(|t| t.as_str().to_string()),
                caps.name("sender").map(|s| s.as_str().trim().to_string()),
            )
        } else {
            (None, None, None)
        };

        let cleaned = clean_block(block, &header_re, &system_re, &emoji_re, &space_re);
        let status = if cleaned.is_empty() { "IGNORED" } else { "NEW" };

        let msg = ParsedMessage {
            date_raw,
            time_raw,
            sender,
            raw_text: block.to_string(),
            cleaned_text: cleaned,
            status: status.to_string(),
        };

        writeln!(output, "{}", serde_json::to_string(&msg)?)?;
    }

    Ok(())
}

fn normalize_whitespace(s: &str, space_re: &Regex) -> String {
    let mut t = s.nfkc().collect::<String>();
    t = t
        .replace('\u{00A0}', " ")
        .replace('\u{202F}', " ")
        .replace('\u{200B}', "")
        .replace('\u{200E}', "")
        .replace('\u{200F}', "");
    space_re.replace_all(&t, " ").to_string()
}

fn clean_block(
    block: &str,
    header_re: &Regex,
    system_re: &Regex,
    emoji_re: &Regex,
    space_re: &Regex,
) -> String {
    if block.is_empty() {
        return String::new();
    }

    let mut cleaned = emoji_re.replace_all(block, " ").to_string();
    cleaned = normalize_whitespace(&cleaned, space_re);

    if let Some(m) = header_re.find(&cleaned) {
        if m.start() == 0 {
            cleaned = cleaned[m.end()..].trim().to_string();
        }
    }

    cleaned
        .lines()
        .filter(|line| !system_re.is_match(line.trim()))
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}
