use anyhow::{Context, Result};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use futures::{stream, StreamExt};
use futures::FutureExt;
use quick_xml::events::Event as XmlEvent;
use quick_xml::Reader;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap};
use ratatui::Terminal;
use regex::Regex;
use reqwest::{header, Client, StatusCode};
use scraper::{Html, Selector};
use std::collections::{HashSet, VecDeque};
use std::fs::OpenOptions;
use std::io::{self, Stdout};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use url::Url;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Max number of Wirecutter page URLs to process (discovered via sitemap)
    #[arg(long, default_value_t = 2000)]
    max_pages: usize,

    /// Concurrency for fetching pages
    #[arg(long, default_value_t = 6)]
    workers: usize,

    /// Follow redirects to final destination
    #[arg(long, default_value_t = false)]
    resolve: bool,

    /// Output CSV path
    #[arg(long, default_value = "wirecutter_amazon_links.csv")]
    out: String,

    /// Optional single URL to process (overrides sitemaps)
    #[arg(long)]
    url: Option<String>,

    /// Add/override sitemap start URLs (can be repeated)
    #[arg(long)]
    sitemap: Vec<String>,

    /// Delay between requests per worker (ms) to be polite
    #[arg(long, default_value_t = 200)]
    delay_ms: u64,

    /// Run without the TUI
    #[arg(long, default_value_t = false)]
    no_tui: bool,
}

#[derive(Clone, Debug)]
struct Row {
    source_page: String,
    wirecutter_link: String,
    final_url: String,
    asin: String,
}

#[derive(Clone, Debug)]
struct Config {
    max_pages: usize,
    workers: usize,
    resolve: bool,
    out: String,
    url: Option<String>,
    sitemaps: Vec<String>,
    delay_ms: u64,
}

impl Config {
    fn from_args(args: &Args) -> Self {
        let sitemaps = if args.sitemap.is_empty() {
            DEFAULT_SITEMAPS.iter().map(|s| s.to_string()).collect()
        } else {
            args.sitemap.clone()
        };
        Self {
            max_pages: args.max_pages,
            workers: args.workers,
            resolve: args.resolve,
            out: args.out.clone(),
            url: args.url.as_ref().map(|u| u.trim().to_string()).filter(|u| !u.is_empty()),
            sitemaps,
            delay_ms: args.delay_ms,
        }
    }
}

static DEFAULT_SITEMAPS: &[&str] = &[
    "https://www.nytimes.com/wirecutter/sitemap.xml",
    "https://www.nytimes.com/wirecutter/sitemap_index.xml",
    "https://www.nytimes.com/sitemaps/new/wirecutter.xml",
    "https://www.nytimes.com/sitemap.xml",
];

const URLS_PATH: &str = "urls.csv";

#[derive(Default, Debug)]
struct SharedProgress {
    total_pages: AtomicUsize,
    processed_pages: AtomicUsize,
    total_links: AtomicUsize,
    written_links: AtomicUsize,
    done: AtomicBool,
    cancelled: AtomicBool,
    failed: AtomicBool,
    logs: Mutex<VecDeque<String>>,
}

impl SharedProgress {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            logs: Mutex::new(VecDeque::with_capacity(200)),
            ..Default::default()
        })
    }

    fn push_log(&self, line: impl Into<String>) {
        let mut logs = self.logs.lock().unwrap();
        if logs.len() >= 200 {
            logs.pop_front();
        }
        logs.push_back(line.into());
    }
}

#[derive(Debug, Clone)]
enum Screen {
    Config,
    Running,
    Done,
    Error,
}

#[derive(Debug)]
struct AppState {
    screen: Screen,
    config: Config,
    urls: Vec<String>,
    sitemap_input: String,
    selected: usize,
    editing: bool,
    input: String,
    status: String,
    error: Option<String>,
    started_at: Option<Instant>,
    progress: Arc<SharedProgress>,
    job: Option<tokio::task::JoinHandle<JobOutcome>>,
}

#[derive(Debug, Clone)]
enum JobOutcome {
    Done { written: usize },
    Cancelled,
    Error(String),
}

#[derive(Debug, Clone, Copy)]
enum Field {
    MaxPages,
    Workers,
    Resolve,
    Output,
    Url,
    Delay,
    Sitemaps,
    Start,
}

const FIELDS: &[Field] = &[
    Field::MaxPages,
    Field::Workers,
    Field::Resolve,
    Field::Output,
    Field::Url,
    Field::Delay,
    Field::Sitemaps,
    Field::Start,
];

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure_urls_csv()?;
    let config = Config::from_args(&args);

    if args.no_tui {
        let progress = SharedProgress::new();
        let outcome = run_job(config, progress).await?;
        match outcome {
            JobOutcome::Done { written } => {
                eprintln!("Done. Wrote {} unique links.", written);
            }
            JobOutcome::Cancelled => {
                eprintln!("Cancelled.");
            }
            JobOutcome::Error(err) => {
                eprintln!("Failed: {}", err);
            }
        }
        return Ok(());
    }

    run_tui(config).await
}

async fn run_tui(config: Config) -> Result<()> {
    let mut terminal = setup_terminal()?;
    let _cleanup = TerminalCleanup;

    let urls = load_urls_from_csv(URLS_PATH).unwrap_or_default();

    let sitemap_input = config.sitemaps.join(", ");
    let mut app = AppState {
        screen: Screen::Config,
        config,
        urls,
        sitemap_input,
        selected: 0,
        editing: false,
        input: String::new(),
        status: "Ready".to_string(),
        error: None,
        started_at: None,
        progress: SharedProgress::new(),
        job: None,
    };

    loop {
        terminal.draw(|frame| draw_ui(frame, &app))?;

        if let Some(handle) = &mut app.job {
            if let Some(outcome) = handle.now_or_never() {
                app.job = None;
                match outcome {
                    Ok(result) => match result {
                        JobOutcome::Done { written } => {
                            app.screen = Screen::Done;
                            app.status = format!("Completed. Wrote {} links.", written);
                        }
                        JobOutcome::Cancelled => {
                            app.screen = Screen::Done;
                            app.status = "Cancelled.".to_string();
                        }
                        JobOutcome::Error(err) => {
                            app.screen = Screen::Error;
                            app.error = Some(err);
                        }
                    },
                    Err(err) => {
                        app.screen = Screen::Error;
                        app.error = Some(format!("Join error: {}", err));
                    }
                }
            }
        }

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if handle_key(&mut app, key)? {
                    break;
                }
            }
        }
    }

    Ok(())
}

fn handle_key(app: &mut AppState, key: KeyEvent) -> Result<bool> {
    match app.screen {
        Screen::Config => handle_key_config(app, key),
        Screen::Running => handle_key_running(app, key),
        Screen::Done | Screen::Error => handle_key_done(app, key),
    }
}

fn handle_key_config(app: &mut AppState, key: KeyEvent) -> Result<bool> {
    if app.editing {
        match key.code {
            KeyCode::Esc => {
                app.editing = false;
                app.input.clear();
                app.status = "Edit cancelled".to_string();
            }
            KeyCode::Enter => {
                apply_edit(app);
            }
            KeyCode::Backspace => {
                app.input.pop();
            }
            KeyCode::Char(c) => {
                if !key.modifiers.contains(KeyModifiers::CONTROL) {
                    app.input.push(c);
                }
            }
            _ => {}
        }
        return Ok(false);
    }

    match key.code {
        KeyCode::Char('q') => return Ok(true),
        KeyCode::Down | KeyCode::Tab => {
            app.selected = (app.selected + 1) % FIELDS.len();
        }
        KeyCode::Up => {
            if app.selected == 0 {
                app.selected = FIELDS.len() - 1;
            } else {
                app.selected -= 1;
            }
        }
        KeyCode::Enter => {
            let field = FIELDS[app.selected];
            match field {
                Field::Resolve => {
                    app.config.resolve = !app.config.resolve;
                    app.status = if app.config.resolve {
                        "Resolve redirects: ON".to_string()
                    } else {
                        "Resolve redirects: OFF".to_string()
                    };
                }
                Field::Url => {
                    if app.urls.is_empty() {
                        app.status = format!("No URLs found in {}", URLS_PATH);
                    } else {
                        app.config.url = Some(next_url_choice(&app.urls, app.config.url.as_ref()));
                        app.status = "URL updated".to_string();
                    }
                }
                Field::Start => start_job(app)?,
                _ => {
                    app.editing = true;
                    app.input = current_field_value(app, field);
                }
            }
        }
        KeyCode::Char(' ') => {
            let field = FIELDS[app.selected];
            if let Field::Resolve = field {
                app.config.resolve = !app.config.resolve;
                app.status = if app.config.resolve {
                    "Resolve redirects: ON".to_string()
                } else {
                    "Resolve redirects: OFF".to_string()
                };
            }
        }
        KeyCode::Char('s') | KeyCode::F(5) => {
            start_job(app)?;
        }
        _ => {}
    }

    Ok(false)
}

fn handle_key_running(app: &mut AppState, key: KeyEvent) -> Result<bool> {
    match key.code {
        KeyCode::Char('q') => {
            app.progress.cancelled.store(true, Ordering::Relaxed);
            app.status = "Cancelling...".to_string();
            Ok(false)
        }
        KeyCode::Esc => Ok(true),
        _ => Ok(false),
    }
}

fn handle_key_done(app: &mut AppState, key: KeyEvent) -> Result<bool> {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => Ok(true),
        KeyCode::Enter => {
            app.screen = Screen::Config;
            app.error = None;
            app.status = "Ready".to_string();
            Ok(false)
        }
        _ => Ok(false),
    }
}

fn start_job(app: &mut AppState) -> Result<()> {
    sync_sitemaps_from_input(app);
    app.progress = SharedProgress::new();
    app.progress.push_log("Starting...".to_string());
    app.status = "Running".to_string();
    app.screen = Screen::Running;
    app.started_at = Some(Instant::now());

    let config = app.config.clone();
    let progress = Arc::clone(&app.progress);
    app.job = Some(tokio::spawn(async move { run_job(config, progress).await.unwrap_or(JobOutcome::Error("Unexpected failure".to_string())) }));
    Ok(())
}

fn sync_sitemaps_from_input(app: &mut AppState) {
    let mut entries: Vec<String> = app
        .sitemap_input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if entries.is_empty() {
        entries = DEFAULT_SITEMAPS.iter().map(|s| s.to_string()).collect();
        app.sitemap_input = entries.join(", ");
    }

    app.config.sitemaps = entries;
}

fn apply_edit(app: &mut AppState) {
    let field = FIELDS[app.selected];
    let input = app.input.trim().to_string();
    let mut ok = true;

    match field {
        Field::MaxPages => match input.parse::<usize>() {
            Ok(v) if v > 0 => app.config.max_pages = v,
            _ => ok = false,
        },
        Field::Workers => match input.parse::<usize>() {
            Ok(v) if v > 0 => app.config.workers = v,
            _ => ok = false,
        },
        Field::Output => {
            if !input.is_empty() {
                app.config.out = input;
            } else {
                ok = false;
            }
        }
        Field::Url => {}
        Field::Delay => match input.parse::<u64>() {
            Ok(v) => app.config.delay_ms = v,
            _ => ok = false,
        },
        Field::Sitemaps => {
            app.sitemap_input = input;
            sync_sitemaps_from_input(app);
        }
        Field::Resolve | Field::Start => {}
    }

    app.editing = false;
    app.input.clear();

    if ok {
        app.status = "Updated".to_string();
        app.error = None;
    } else {
        app.status = "Invalid value".to_string();
        app.error = Some("Invalid value for field".to_string());
    }
}

fn current_field_value(app: &AppState, field: Field) -> String {
    match field {
        Field::MaxPages => app.config.max_pages.to_string(),
        Field::Workers => app.config.workers.to_string(),
        Field::Resolve => if app.config.resolve { "true" } else { "false" }.to_string(),
        Field::Output => app.config.out.clone(),
        Field::Url => app.config.url.clone().unwrap_or_default(),
        Field::Delay => app.config.delay_ms.to_string(),
        Field::Sitemaps => app.sitemap_input.clone(),
        Field::Start => String::new(),
    }
}

fn draw_ui(frame: &mut ratatui::Frame, app: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(frame.size());

    let title = Paragraph::new("Wirecutter Amazon Link Extractor")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center);
    frame.render_widget(title, chunks[0]);

    match app.screen {
        Screen::Config => draw_config(frame, app, chunks[1]),
        Screen::Running => draw_running(frame, app, chunks[1]),
        Screen::Done => draw_done(frame, app, chunks[1]),
        Screen::Error => draw_error(frame, app, chunks[1]),
    }

    let footer = Paragraph::new(Text::from(Line::from(vec![
        Span::styled("Status: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(&app.status),
        Span::raw("  "),
        Span::styled("q", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(": quit"),
    ])))
    .block(Block::default().borders(Borders::TOP));
    frame.render_widget(footer, chunks[2]);
}

fn draw_config(frame: &mut ratatui::Frame, app: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(area);

    let items = FIELDS.iter().enumerate().map(|(i, field)| {
        let selected = i == app.selected;
        let mut style = Style::default();
        if selected {
            style = style.fg(Color::Yellow).add_modifier(Modifier::BOLD);
        }
        let label = match field {
            Field::MaxPages => "Max pages",
            Field::Workers => "Workers",
            Field::Resolve => "Resolve redirects",
            Field::Output => "Output CSV",
            Field::Url => "URL",
            Field::Delay => "Delay ms",
            Field::Sitemaps => "Sitemaps",
            Field::Start => "Start",
        };
        let value = match field {
            Field::MaxPages => app.config.max_pages.to_string(),
            Field::Workers => app.config.workers.to_string(),
            Field::Resolve => if app.config.resolve { "ON" } else { "OFF" }.to_string(),
            Field::Output => app.config.out.clone(),
            Field::Url => app.config.url.clone().unwrap_or_else(|| "(none)".to_string()),
            Field::Delay => app.config.delay_ms.to_string(),
            Field::Sitemaps => app.sitemap_input.clone(),
            Field::Start => "[ Press Enter or S ]".to_string(),
        };
        let line = format!("{:<18} {}", label, value);
        ListItem::new(Line::from(line)).style(style)
    });

    let list = List::new(items)
        .block(Block::default().title("Configuration").borders(Borders::ALL));
    frame.render_widget(list, chunks[0]);

    let help = Paragraph::new("Arrows/Tab to move. Enter to edit. URL field cycles choices. Space toggles resolve. S or F5 to start.")
        .alignment(Alignment::Left)
        .block(Block::default().borders(Borders::TOP))
        .wrap(Wrap { trim: true });
    frame.render_widget(help, chunks[1]);

    if app.editing {
        let edit_area = centered_rect(70, 30, frame.size());
        let block = Block::default().title("Edit value").borders(Borders::ALL);
        let input = Paragraph::new(app.input.as_str())
            .block(block)
            .alignment(Alignment::Left);
        frame.render_widget(input, edit_area);
    }
}

fn draw_running(frame: &mut ratatui::Frame, app: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(8),
        ])
        .split(area);

    let total = app.progress.total_pages.load(Ordering::Relaxed);
    let processed = app.progress.processed_pages.load(Ordering::Relaxed);
    let pct = if total == 0 { 0.0 } else { (processed as f64 / total as f64).min(1.0) };
    let label = format!("{}/{} pages", processed, total);

    let gauge = Gauge::default()
        .block(Block::default().title("Progress").borders(Borders::ALL))
        .gauge_style(Style::default().fg(Color::Green))
        .ratio(pct)
        .label(label);
    frame.render_widget(gauge, chunks[0]);

    let mut lines = Vec::new();
    let total_links = app.progress.total_links.load(Ordering::Relaxed);
    let written = app.progress.written_links.load(Ordering::Relaxed);
    let elapsed = app.started_at.map(|s| s.elapsed().as_secs()).unwrap_or(0);
    lines.push(Line::from(format!("Links found: {}", total_links)));
    lines.push(Line::from(format!("Links written: {}", written)));
    lines.push(Line::from(format!("Elapsed: {}s", elapsed)));

    let logs = app.progress.logs.lock().unwrap();
    let log_items: Vec<ListItem> = logs.iter().rev().take(8).rev().map(|l| ListItem::new(Line::from(l.clone()))).collect();
    drop(logs);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(5),
        ])
        .split(chunks[1]);

    let stats = Paragraph::new(Text::from(lines))
        .block(Block::default().title("Stats").borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    frame.render_widget(stats, layout[0]);

    let logs_widget = List::new(log_items)
        .block(Block::default().title("Recent activity").borders(Borders::ALL));
    frame.render_widget(logs_widget, layout[1]);
}

fn draw_done(frame: &mut ratatui::Frame, app: &AppState, area: Rect) {
    let text = format!("{}\n\nPress Enter to configure again, or Q to quit.", app.status);
    let para = Paragraph::new(text)
        .block(Block::default().title("Finished").borders(Borders::ALL))
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });
    frame.render_widget(para, area);
}

fn draw_error(frame: &mut ratatui::Frame, app: &AppState, area: Rect) {
    let message = app
        .error
        .clone()
        .unwrap_or_else(|| "Unknown error".to_string());
    let text = format!("{}\n\nPress Enter to return to config, or Q to quit.", message);
    let para = Paragraph::new(text)
        .block(Block::default().title("Error").borders(Borders::ALL))
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });
    frame.render_widget(para, area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

struct TerminalCleanup;

impl Drop for TerminalCleanup {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen);
    }
}

async fn run_job(config: Config, progress: Arc<SharedProgress>) -> Result<JobOutcome> {
    let client = build_client()?;
    let sitemaps = config.sitemaps.clone();

    let asin_re = Regex::new(r"/(?:dp|gp/product)/([A-Z0-9]{10})(?:[/?]|$)")?;
    let link_sel = Selector::parse("a[href]").unwrap();

    let (mut existing, existing_pages) = load_existing_keys_and_pages(&config.out).unwrap_or_default();
    if !existing.is_empty() {
        progress.push_log(format!("Resuming: {} existing URLs in {}", existing.len(), config.out));
    }
    if !existing_pages.is_empty() {
        progress.push_log(format!(
            "Resuming: {} source pages already processed",
            existing_pages.len()
        ));
    }
    let writer = open_csv_writer(&config.out)?;

    let pages = if let Some(url) = &config.url {
        progress.push_log(format!("Using URL from {}: {}", URLS_PATH, url));
        vec![url.clone()]
    } else {
        discover_pages_from_sitemaps(
            &client,
            &sitemaps,
            config.max_pages,
            &existing_pages,
            &progress,
        )
        .await
        .context("discovering pages from sitemaps")?
    };

    if pages.is_empty() {
        return Ok(JobOutcome::Error(format!(
            "No new pages found. Try increasing max pages or clearing {}",
            config.out
        )));
    }

    progress.total_pages.store(pages.len(), Ordering::Relaxed);
    progress.push_log(format!(
        "Discovered {} pages. Starting with {} workers.",
        pages.len(),
        config.workers
    ));

    let total_pages = pages.len();
    let processed_pages = &progress.processed_pages;
    let total_links = &progress.total_links;
    let written_links = &progress.written_links;
    let seen = Arc::new(tokio::sync::Mutex::new(std::mem::take(&mut existing)));
    let writer = Arc::new(tokio::sync::Mutex::new(writer));
    let existing_pages = Arc::new(existing_pages);

    stream::iter(pages.into_iter())
        .map(|page_url| {
            let client = client.clone();
            let asin_re = asin_re.clone();
            let link_sel = link_sel.clone();
            let resolve = config.resolve;
            let delay = Duration::from_millis(config.delay_ms);
            let processed_pages = processed_pages.clone();
            let total_links = total_links.clone();
            let written_links = written_links.clone();
            let seen = Arc::clone(&seen);
            let writer = Arc::clone(&writer);
            let existing_pages = Arc::clone(&existing_pages);
            let progress = Arc::clone(&progress);

            async move {
                if progress.cancelled.load(Ordering::Relaxed) {
                    return;
                }

                if existing_pages.contains(&page_url) {
                    let page_count = processed_pages.fetch_add(1, Ordering::Relaxed) + 1;
                    if page_count % 100 == 0 || page_count == total_pages {
                        let links = total_links.load(Ordering::Relaxed);
                        let written = written_links.load(Ordering::Relaxed);
                        progress.push_log(format!(
                            "Processed {}/{} pages, {} links found, {} written",
                            page_count, total_pages, links, written
                        ));
                    }
                    return;
                }

                sleep(delay).await;

                let rows = match process_page(&client, &page_url, &link_sel, resolve, &asin_re).await {
                    Ok(r) => r,
                    Err(_) => Vec::new(),
                };

                if !rows.is_empty() {
                    total_links.fetch_add(rows.len(), Ordering::Relaxed);
                    let mut unique = Vec::new();
                    let mut seen_guard = seen.lock().await;
                    for r in rows {
                        let key = if !r.final_url.is_empty() {
                            r.final_url.clone()
                        } else {
                            r.wirecutter_link.clone()
                        };
                        if seen_guard.insert(key) {
                            unique.push(r);
                        }
                    }
                    drop(seen_guard);

                    if !unique.is_empty() {
                        let mut w = writer.lock().await;
                        for r in &unique {
                            let _ = w.write_record([
                                &r.source_page,
                                &r.wirecutter_link,
                                &r.final_url,
                                &r.asin,
                            ]);
                        }
                        let _ = w.flush();
                        written_links.fetch_add(unique.len(), Ordering::Relaxed);
                    }
                }

                let page_count = processed_pages.fetch_add(1, Ordering::Relaxed) + 1;
                if page_count % 100 == 0 || page_count == total_pages {
                    let links = total_links.load(Ordering::Relaxed);
                    let written = written_links.load(Ordering::Relaxed);
                    progress.push_log(format!(
                        "Processed {}/{} pages, {} links found, {} written",
                        page_count, total_pages, links, written
                    ));
                }
            }
        })
        .buffer_unordered(config.workers)
        .for_each(|_| async {})
        .await;

    progress.done.store(true, Ordering::Relaxed);

    if progress.cancelled.load(Ordering::Relaxed) {
        return Ok(JobOutcome::Cancelled);
    }

    let written = progress.written_links.load(Ordering::Relaxed);
    Ok(JobOutcome::Done { written })
}

fn build_client() -> Result<Client> {
    let mut headers = header::HeaderMap::new();
    headers.insert(
        header::USER_AGENT,
        header::HeaderValue::from_static("Mozilla/5.0 (compatible; link-auditor/1.0)"),
    );

    let client = Client::builder()
        .default_headers(headers)
        .cookie_store(true)
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    Ok(client)
}

async fn discover_pages_from_sitemaps(
    client: &Client,
    starts: &[String],
    max_pages: usize,
    skip_pages: &HashSet<String>,
    progress: &SharedProgress,
) -> Result<Vec<String>> {
    let mut seen_sitemaps: HashSet<String> = HashSet::new();
    let mut seen_pages: HashSet<String> = HashSet::new();
    let mut q: VecDeque<String> = starts.iter().cloned().collect();
    let mut processed_sitemaps: usize = 0;
    let mut last_page_log: usize = 0;

    progress.push_log(format!(
        "Discovering pages from {} sitemaps (max_pages={})",
        starts.len(),
        max_pages
    ));

    while let Some(sm_url) = q.pop_front() {
        if progress.cancelled.load(Ordering::Relaxed) {
            break;
        }
        if seen_pages.len() >= max_pages {
            break;
        }
        if !seen_sitemaps.insert(sm_url.clone()) {
            continue;
        }

        let resp = match client.get(&sm_url).send().await {
            Ok(r) => r,
            Err(_) => continue,
        };

        let status = resp.status();
        if status != StatusCode::OK {
            continue;
        }

        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(_) => continue,
        };

        let (page_urls, sitemap_urls) = parse_sitemap(&sm_url, &bytes)?;

        for u in sitemap_urls {
            if !seen_sitemaps.contains(&u) {
                q.push_back(u);
            }
        }
        for u in page_urls {
            if seen_pages.len() >= max_pages {
                break;
            }
            if skip_pages.contains(&u) {
                continue;
            }
            seen_pages.insert(u);
        }

        processed_sitemaps += 1;
        if processed_sitemaps % 5 == 0
            || seen_pages.len() >= max_pages
            || seen_pages.len() >= last_page_log + 500
        {
            progress.push_log(format!(
                "Discovery progress: sitemaps={}, pages={}, queue={}",
                processed_sitemaps,
                seen_pages.len(),
                q.len()
            ));
            last_page_log = seen_pages.len();
        }
    }

    Ok(seen_pages.into_iter().collect())
}

fn parse_sitemap(base: &str, xml: &[u8]) -> Result<(Vec<String>, Vec<String>)> {
    let mut reader = Reader::from_reader(xml);
    reader.trim_text(true);

    let mut buf = Vec::new();
    let mut in_loc = false;
    let mut locs: Vec<String> = Vec::new();

    let mut saw_urlset = false;
    let mut saw_sitemapindex = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(XmlEvent::Start(e)) => {
                if e.name().as_ref().ends_with(b"urlset") {
                    saw_urlset = true;
                } else if e.name().as_ref().ends_with(b"sitemapindex") {
                    saw_sitemapindex = true;
                } else if e.name().as_ref().ends_with(b"loc") {
                    in_loc = true;
                }
            }
            Ok(XmlEvent::End(e)) => {
                if e.name().as_ref().ends_with(b"loc") {
                    in_loc = false;
                }
            }
            Ok(XmlEvent::Text(t)) => {
                if in_loc {
                    let text = t.unescape()?.to_string();
                    locs.push(text);
                }
            }
            Ok(XmlEvent::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    let base_url = Url::parse(base).ok();
    let norm = |u: String| -> String {
        if let Ok(parsed) = Url::parse(&u) {
            parsed.to_string()
        } else if let Some(b) = &base_url {
            b.join(&u).map(|x| x.to_string()).unwrap_or(u)
        } else {
            u
        }
    };

    if saw_sitemapindex && !saw_urlset {
        let sitemaps = locs.into_iter().map(norm).collect();
        Ok((Vec::new(), sitemaps))
    } else {
        let pages = locs.into_iter().map(norm).collect();
        Ok((pages, Vec::new()))
    }
}

async fn process_page(
    client: &Client,
    page_url: &str,
    link_sel: &Selector,
    resolve: bool,
    asin_re: &Regex,
) -> Result<Vec<Row>> {
    let resp = client.get(page_url).send().await?;
    if resp.status() != StatusCode::OK {
        return Ok(Vec::new());
    }
    let html = resp.text().await?;
    let candidates = extract_candidate_links(&html, page_url, link_sel);

    let mut rows = Vec::new();
    for c in candidates {
        let final_url = if resolve {
            resolve_final(client, &c).await.unwrap_or_default()
        } else {
            String::new()
        };

        let asin = asin_from_url(asin_re, if final_url.is_empty() { &c } else { &final_url });

        rows.push(Row {
            source_page: page_url.to_string(),
            wirecutter_link: c,
            final_url,
            asin,
        });
    }
    Ok(rows)
}

fn extract_candidate_links(html: &str, page_url: &str, sel: &Selector) -> HashSet<String> {
    let doc = Html::parse_document(html);
    let mut out = HashSet::new();

    let base = Url::parse(page_url).ok();

    for a in doc.select(sel) {
        let href = match a.value().attr("href") {
            Some(h) => h.trim(),
            None => continue,
        };

        let abs = match Url::parse(href) {
            Ok(u) => u,
            Err(_) => match (&base, href) {
                (Some(b), rel) => b.join(rel).unwrap_or_else(|_| return_url_fallback(rel)),
                _ => return_url_fallback(href),
            },
        };

        let host = abs.host_str().unwrap_or("").to_lowercase();
        let path = abs.path().to_string();
        let full = abs.to_string();

        if host.contains("amazon.") {
            out.insert(full);
            continue;
        }

        if host.contains("nytimes.com") && path.contains("/wirecutter/out/") {
            let q = abs.query().unwrap_or("").to_lowercase();
            if q.contains("merchant=amazon") {
                out.insert(full);
            }
        }
    }

    out
}

fn return_url_fallback(s: &str) -> Url {
    Url::parse("https://invalid.local/")
        .unwrap()
        .join(s)
        .unwrap_or_else(|_| Url::parse("https://invalid.local/").unwrap())
}

async fn resolve_final(client: &Client, url: &str) -> Result<String> {
    let head = client.head(url).send().await;
    if let Ok(r) = head {
        if r.status().is_success() {
            return Ok(r.url().to_string());
        }
    }

    let get = client.get(url).send().await?;
    Ok(get.url().to_string())
}

fn asin_from_url(asin_re: &Regex, url: &str) -> String {
    asin_re
        .captures(url)
        .and_then(|c| c.get(1).map(|m| m.as_str().to_string()))
        .unwrap_or_default()
}

fn open_csv_writer(path: &str) -> Result<csv::Writer<std::fs::File>> {
    let exists = Path::new(path).exists();
    let size = if exists {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut w = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);

    if size == 0 {
        w.write_record(["source_wirecutter_page", "wirecutter_link", "final_url", "asin"])?;
        w.flush()?;
    }

    Ok(w)
}

fn load_existing_keys_and_pages(path: &str) -> Result<(HashSet<String>, HashSet<String>)> {
    if !Path::new(path).exists() {
        return Ok((HashSet::new(), HashSet::new()));
    }

    let mut rdr = csv::ReaderBuilder::new().from_path(path)?;
    let mut seen_links = HashSet::new();
    let mut seen_pages = HashSet::new();
    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        let source_page = record.get(0).unwrap_or("").to_string();
        let wirecutter_link = record.get(1).unwrap_or("").to_string();
        let final_url = record.get(2).unwrap_or("").to_string();
        let key = if !final_url.is_empty() {
            final_url
        } else {
            wirecutter_link
        };
        if !key.is_empty() {
            seen_links.insert(key);
        }
        if !source_page.is_empty() {
            seen_pages.insert(source_page);
        }
    }

    Ok((seen_links, seen_pages))
}

fn ensure_urls_csv() -> Result<()> {
    if Path::new(URLS_PATH).exists() {
        return Ok(());
    }

    let file = OpenOptions::new().create(true).write(true).open(URLS_PATH)?;
    let mut w = csv::WriterBuilder::new().has_headers(false).from_writer(file);
    w.write_record(["url"])?;
    w.flush()?;
    Ok(())
}

fn load_urls_from_csv(path: &str) -> Result<Vec<String>> {
    if !Path::new(path).exists() {
        return Ok(Vec::new());
    }

    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(path)?;
    let mut urls = Vec::new();
    let mut seen = HashSet::new();
    let mut skipped_header = false;
    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        let value = record.get(0).unwrap_or("").trim();
        if value.is_empty() {
            continue;
        }
        if !skipped_header && value.eq_ignore_ascii_case("url") {
            skipped_header = true;
            continue;
        }
        if seen.insert(value.to_string()) {
            urls.push(value.to_string());
        }
    }

    Ok(urls)
}

fn next_url_choice(urls: &[String], current: Option<&String>) -> String {
    if urls.is_empty() {
        return String::new();
    }

    if let Some(curr) = current {
        if let Some(pos) = urls.iter().position(|u| u == curr) {
            let next = (pos + 1) % urls.len();
            return urls[next].clone();
        }
    }

    urls[0].clone()
}
