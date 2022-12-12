use clap::Parser;
use config::Config;
use std::ffi::OsString;
use std::process::Command;

#[derive(Parser)] // requires `derive` feature
#[command(name = "command")]
enum CliCommand {
    #[command(short_flag = 'e')]
    Execute(ExecuteArgs),
    #[command(short_flag = 'i')]
    Interactive,
}

#[derive(clap::Args)]
#[command(author, version, about, long_about = None)]
struct ExecuteArgs {
    commandline: Vec<OsString>,
}

fn main() -> Result<(), ::std::io::Error> {
    let settings = Config::builder()
        .add_source(config::File::with_name("Robocom"))
        .add_source(config::Environment::with_prefix("ROBOCOM"))
        .build()
        .unwrap();
    println!("settings: {:?}", settings);
    let multicast = settings.get_table("multicast").unwrap();
    let group = multicast.get("group").unwrap();
    println!("group: {:?}", group);

    match CliCommand::parse() {
        CliCommand::Execute(args) => {
            if let Some(exec) = args.commandline.get(0) {
                let cmd = Command::new(exec);
                println!("Executable: {:?}", cmd.get_program());
                Ok(())
            } else {
                println!("{:?}", args.commandline.get(0));
                Ok(())
            }
        }
        _ => {
            println!("...todo");
            Ok(())
        }
    }
}
