//! cr_wrap
#![deny(missing_docs)]

pub mod arc;
pub mod chemistry_arg;
pub mod cloud;
pub mod create_bam_arg;
mod deprecated_os;
pub mod env;
pub mod fastqs;
pub mod mkfastq;
pub mod mkref;
pub mod mrp_args;
pub mod shared_cmd;
pub mod telemetry;
pub mod utils;

use anyhow::{ensure, Context, Result};
use mrp_args::MrpArgs;
use serde::Serialize;
use std::fs::{create_dir, File};
use std::io::Write;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus, Stdio};
use std::time::Instant;
use telemetry::TelemetryCollector;

/// Convert something to an ExitCode.
trait IntoExitCode {
    fn into_u8(self) -> u8;
    fn into_exit_code(self) -> ExitCode;
}

impl IntoExitCode for ExitStatus {
    /// Return the exit code, or 128 + the signal number, or 255.
    fn into_u8(self) -> u8 {
        if let Some(code) = self.code() {
            code as u8
        } else if let Some(signal) = self.signal() {
            128 + signal as u8
        } else {
            255
        }
    }

    /// Convert an ExitStatus to an ExitCode.
    fn into_exit_code(self) -> ExitCode {
        ExitCode::from(self.into_u8())
    }
}

/// Generate an MRO invocation string
/// Args:
///  - `call`: the pipeline to invoke.
///  - `args`: the pipeline argument - will be serialized to json and passed to mrg
///  - `pipeline_mro_file`: the MRO file declaring the pipeline. Must be on the MROPATH.
pub fn make_mro<T: Serialize>(call: &str, args: &T, pipeline_mro_file: &str) -> Result<String> {
    let args =
        serde_json::to_value(args).with_context(|| "error serializing pipeline args to json")?;

    let json = serde_json::json!({
        "call": call,
        "args": args,
        "mro_file": pipeline_mro_file,
    });

    let msg = serde_json::to_string_pretty(&json)?;

    let mro_string =
        call_mrg(&msg).with_context(|| format!("failure calling mrg on json:\n {msg}"))?;
    Ok(mro_string)
}

/// Make an MRO file with a comment
pub fn make_mro_with_comment<T: Serialize>(
    call: &str,
    args: &T,
    pipeline_mro_file: &str,
    comment: &str,
) -> Result<String> {
    let comment_with_hashes: String = comment
        .lines()
        .map(|line| "# ".to_string() + line + "\n")
        .collect();
    Ok(comment_with_hashes + &make_mro(call, args, pipeline_mro_file)?)
}

fn call_mrg(msg: &str) -> Result<String> {
    let mut child = Command::new("mrg")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .with_context(|| "Failed to run mrg")?;

    {
        let stdin = child.stdin.as_mut().expect("Failed to open stdin of mrg");
        stdin
            .write_all(msg.as_bytes())
            .with_context(|| "Failed to write to stdin of mrg")?;
    }

    let invocation = child
        .wait_with_output()
        .with_context(|| "Failed to read stdout of mrg")?;
    let result = String::from_utf8(invocation.stdout)?;

    ensure!(
        invocation.status.success(),
        "Creating MRO invocation: {result}"
    );

    Ok(result)
}

#[allow(missing_docs)]
pub enum MroInvocation {
    MroString(String),
    File(PathBuf),
}

/// Execute a Martian pipeline with mrp and return an ExitCode.
/// Args:
///  - `job_id` the job name. Job results will be written to this directory.
///  - `invocation`: string of the MRO invocation to run.
///  - `mrg_args`: additional parameters to pass to mrp controlling the job execution
///  - `dry_run`: emit the MRO that would be run but don't invoke with mrp.
pub fn execute(
    job_id: &str,
    invocation: &str,
    mrp_args: &MrpArgs,
    dry_run: bool,
    telemetry: &mut TelemetryCollector,
) -> Result<ExitCode> {
    Ok(execute_to_status(job_id, invocation, mrp_args, dry_run, telemetry)?.into_exit_code())
}

/// Execute a Martian pipeline with mrp and return an ExitStatus.
pub fn execute_to_status(
    job_id: &str,
    invocation: &str,
    mrp_args: &MrpArgs,
    dry_run: bool,
    telemetry: &mut TelemetryCollector,
) -> Result<ExitStatus> {
    let inv = MroInvocation::MroString(invocation.to_string());
    let start = Instant::now();
    let mrp_exit_status = execute_any(job_id, inv, mrp_args, dry_run)?;
    telemetry.collect(
        Some(&format!("{} {job_id}", mrp_args.get_args().join(" "))),
        Some(&mrp_exit_status),
        Some(start.elapsed()),
    );
    Ok(mrp_exit_status)
}

fn execute_any(
    job_id: &str,
    invocation: MroInvocation,
    mrp_args: &MrpArgs,
    dry_run: bool,
) -> Result<ExitStatus> {
    let (mro_file, tmp) = match invocation {
        MroInvocation::MroString(mro) => {
            let filename: PathBuf = format!("__{job_id}.mro").into();
            let mut f = File::create(&filename).with_context(|| "couldn't open MRO file")?;
            f.write_all(mro.as_bytes())?;
            (filename, true)
        }
        MroInvocation::File(f) => (f, false),
    };

    if dry_run {
        println!("Dry Run Mode");
        println!();
        println!(
            "mrp command: {:?} {} {}",
            mro_file,
            job_id,
            mrp_args.get_args().join(" ")
        );
        println!("mro file: {mro_file:?}");

        return Ok(Command::new("mro")
            .arg("check")
            .arg(mro_file)
            .output()?
            .status);
    }

    let exit_status = run_mrp(job_id, &mro_file, mrp_args)?;
    if tmp {
        std::fs::remove_file(mro_file)?;
    }
    let output_dir = mrp_args.output_dir.as_deref().unwrap_or(job_id);
    let _ = run_tarmri(output_dir, exit_status)?;
    Ok(exit_status)
}

fn run_mrp(job_id: &str, mro_path: &Path, mrp_args: &MrpArgs) -> Result<ExitStatus> {
    if let Some(output_dir) = &mrp_args.output_dir {
        // Create output_dir to ensure that it is created successfully.
        match create_dir(output_dir) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(error) => return Err(error).context(output_dir.clone()),
        }
    };

    let args = mrp_args.get_args();
    Command::new("mrp")
        .arg(mro_path)
        .arg(job_id)
        .args(&args)
        .status()
        .context(format!(
            "running mrp {} {job_id} {}",
            mro_path.display(),
            args.join(" ")
        ))
}

fn run_tarmri(output_dir: &str, exit_status: ExitStatus) -> Result<ExitCode> {
    let mut cmd = Command::new("tarmri");

    // mro and output path
    cmd.arg(output_dir);
    cmd.arg(exit_status.code().unwrap_or(1).to_string());

    let cmdline: Vec<String> = std::env::args().collect();
    let cmdline = cmdline.join(" ");
    cmd.arg(cmdline);

    Ok(cmd
        .status()
        .with_context(|| "Error reading from tarmri")?
        .into_exit_code())
}

/// Check for a deprecated OS
pub fn check_deprecated_os() -> Result<()> {
    if std::env::var("TENX_IGNORE_DEPRECATED_OS")
        .map(|s| s != "0")
        .unwrap_or(false)
    {
        return Ok(());
    }
    // by running this check in a separate subprocess we can do so inside a
    // sanitized environment sans e.g. LD_PRELOAD or LD_LIBRARY_PATH,
    // which could render these checks unreliable
    // if we're running `cellranger oscheck`, don't recurse
    let output = Command::new(std::env::current_exe()?)
        .arg("oscheck")
        .env("TENX_IGNORE_DEPRECATED_OS", "1")
        .output()
        .expect("failed to run oscheck");

    let msg = format!(
        "{}\n{}\n",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    ensure!(output.status.success(), "{}", msg.trim_end_matches('\n'));
    eprint!("{msg}");
    Ok(())
}
