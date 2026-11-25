//! The `rmda` binary: the command line of [`rmda_cli`].

fn main() {
    rmda_cli::main_with_args(std::env::args_os())
}
