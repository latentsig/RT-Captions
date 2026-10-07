fn main() {
    if let Err(e) = rt_captions_lib::cli::run(std::env::args().skip(1).collect()) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
