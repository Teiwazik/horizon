//! The Java runtime that prepares the game files on the device, where none is installed (a
//! phone): OpenJDK 25 built for Android by AngelAuraMC, the build Amethyst runs (GPLv2 with the
//! Classpath Exception; sources and build scripts at github.com/AngelAuraMC/angelauramc-openjdk-build).
//! It is fetched once, checked against the hash pinned here, and unpacked in a folder of its own;
//! the preparer runs Mojang's code in it (horizon-prepare's exporters), and it can go afterwards.
use crate::files::Failure;
use sha2::{Digest, Sha256};
use std::io::{BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};

/// A runtime archive (`.tar.xz`) and what it must be.
pub struct Runtime {
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// OpenJDK 25.0.5 for Android on 64-bit ARM (its release `download_jre25`).
pub const JRE25_ANDROID_ARM64: Runtime = Runtime {
    url: "https://github.com/AngelAuraMC/angelauramc-openjdk-build/releases/download/download_jre25/jre25-android-arm64.tar.xz",
    sha256: "d3eb7afe2240c26728a1bb440502c5f18ac3883e932d202dd7f0c9bcbbce4c37",
    size: 38_031_580,
};

/// Written in the runtime's folder once it is unpacked: the archive's hash.
const INSTALLED: &str = ".starling-runtime";

/// Whether `runtime` is unpacked in `dir`.
pub fn installed(runtime: &Runtime, dir: &Path) -> bool {
    std::fs::read_to_string(dir.join(INSTALLED)).is_ok_and(|s| s.trim() == runtime.sha256)
}

/// `runtime` in `dir` (its `bin`, `lib`...), fetched and unpacked unless it is there already.
/// `progress` gets the bytes fetched and the archive's size; `cancelled` stops the download.
pub fn install(
    runtime: &Runtime,
    dir: &Path,
    progress: &dyn Fn(u64, u64),
    cancelled: &dyn Fn() -> bool,
) -> Result<(), String> {
    if installed(runtime, dir) {
        return Ok(());
    }
    let parent = dir.parent().ok_or("the runtime's folder has no parent")?;
    std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    let archive = dir.with_extension("tar.xz");
    fetch(runtime, &archive, progress, cancelled)?;
    let unpacked = unpack(&archive, dir);
    let _ = std::fs::remove_file(&archive);
    unpacked?;
    std::fs::write(dir.join(INSTALLED), runtime.sha256).map_err(|e| e.to_string())
}

/// The archive into `path`, through a `.part` file, checked as it streams; tried again from the
/// start when the network drops ([`crate::files::retrying`]).
fn fetch(
    runtime: &Runtime,
    path: &Path,
    progress: &dyn Fn(u64, u64),
    cancelled: &dyn Fn() -> bool,
) -> Result<(), String> {
    crate::files::retrying(|| fetch_once(runtime, path, progress, cancelled))
}

fn fetch_once(
    runtime: &Runtime,
    path: &Path,
    progress: &dyn Fn(u64, u64),
    cancelled: &dyn Fn() -> bool,
) -> Result<(), Failure> {
    const NAME: &str = "the Java runtime";
    let response = crate::files::agent()
        .get(runtime.url)
        .call()
        .map_err(|e| match e {
            ureq::Error::Status(code, _) => {
                let e = format!("GitHub refused {NAME} ({code}).");
                if code >= 500 {
                    Failure::Again(e)
                } else {
                    Failure::Final(e)
                }
            }
            ureq::Error::Transport(t) => Failure::Again(format!(
                "Could not download {NAME} ({}). Check your internet connection.",
                crate::transport_detail(&t)
            )),
        })?;
    let disk = |e: String| Failure::Final(e);
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let mut file =
        std::fs::File::create(&part).map_err(|e| disk(format!("{}: {e}", part.display())))?;
    let mut reader = response.into_reader();
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1 << 16];
    let mut fetched = 0u64;
    loop {
        if cancelled() {
            drop(file);
            let _ = std::fs::remove_file(&part);
            return Err(Failure::Final("Stopped.".into()));
        }
        let n = reader
            .read(&mut buffer)
            .map_err(|_| Failure::Again(format!("The download of {NAME} was cut off.")))?;
        if n == 0 {
            break;
        }
        fetched += n as u64;
        if fetched > runtime.size {
            break;
        }
        hasher.update(&buffer[..n]);
        file.write_all(&buffer[..n])
            .map_err(|e| disk(format!("{}: {e}", part.display())))?;
        progress(fetched, runtime.size);
    }
    drop(file);
    // Short: cut off on the way, worth another try. Whole but another file: never.
    if fetched < runtime.size {
        let _ = std::fs::remove_file(&part);
        return Err(Failure::Again(format!(
            "The download of {NAME} was cut off."
        )));
    }
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if fetched != runtime.size || digest != runtime.sha256 {
        let _ = std::fs::remove_file(&part);
        return Err(Failure::Final(format!(
            "{NAME} is not the file Starling expects (its hash differs)."
        )));
    }
    std::fs::rename(&part, path).map_err(|e| disk(format!("{}: {e}", path.display())))
}

/// A `.tar.xz` into `dir`, through a sibling folder renamed once whole. Entries stay inside it;
/// links are made where the system has them (they are only the licence files' copies).
pub fn unpack(archive: &Path, dir: &Path) -> Result<(), String> {
    let mut staging = dir.as_os_str().to_owned();
    staging.push(".unpacking");
    let staging = PathBuf::from(staging);
    for old in [&staging, &dir.to_path_buf()] {
        if old.exists() {
            std::fs::remove_dir_all(old).map_err(|e| format!("{}: {e}", old.display()))?;
        }
    }
    std::fs::create_dir_all(&staging).map_err(|e| format!("{}: {e}", staging.display()))?;
    let tar = staging.with_extension("tar");
    let result = (|| {
        {
            let mut input = BufReader::new(
                std::fs::File::open(archive).map_err(|e| format!("{}: {e}", archive.display()))?,
            );
            let mut output = std::io::BufWriter::new(
                std::fs::File::create(&tar).map_err(|e| format!("{}: {e}", tar.display()))?,
            );
            lzma_rs::xz_decompress(&mut input, &mut output)
                .map_err(|e| format!("The Java runtime's archive is damaged: {e}"))?;
            output.flush().map_err(|e| e.to_string())?;
        }
        let file = std::fs::File::open(&tar).map_err(|e| format!("{}: {e}", tar.display()))?;
        let mut entries = tar::Archive::new(file);
        for entry in entries.entries().map_err(|e| e.to_string())? {
            let mut entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path().map_err(|e| e.to_string())?.into_owned();
            if path
                .components()
                .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
            {
                return Err(format!(
                    "The Java runtime's archive names {}",
                    path.display()
                ));
            }
            if entry.header().entry_type().is_symlink() && !cfg!(unix) {
                continue;
            }
            entry
                .unpack_in(&staging)
                .map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(&tar);
    if let Err(e) = result {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }
    std::fs::rename(&staging, dir).map_err(|e| format!("{}: {e}", dir.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("starling-java-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn archive(dir: &Path, entries: &[(&str, &[u8])], link: Option<(&str, &str)>) -> PathBuf {
        let mut builder = tar::Builder::new(Vec::new());
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, *data).unwrap();
        }
        if let Some((name, target)) = link {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(tar::EntryType::Symlink);
            header.set_size(0);
            builder.append_link(&mut header, name, target).unwrap();
        }
        let tar = builder.into_inner().unwrap();
        let mut xz = Vec::new();
        lzma_rs::xz_compress(&mut &tar[..], &mut xz).unwrap();
        let path = dir.join("runtime.tar.xz");
        std::fs::write(&path, xz).unwrap();
        path
    }

    #[test]
    fn a_runtime_unpacks_whole_into_its_folder() {
        let dir = temp("unpack");
        let archive = archive(
            &dir,
            &[
                ("./release", b"JAVA_VERSION=\"25\"\n"),
                ("./lib/libjli.so", b"elf"),
                ("./legal/java.base/LICENSE", b"GPL"),
            ],
            Some(("./legal/jdk.net/LICENSE", "../java.base/LICENSE")),
        );
        let jre = dir.join("jre");
        std::fs::create_dir_all(jre.join("stale")).unwrap();
        unpack(&archive, &jre).unwrap();
        assert_eq!(std::fs::read(jre.join("lib/libjli.so")).unwrap(), b"elf");
        assert!(
            !jre.join("stale").exists(),
            "an earlier unpacking is replaced"
        );
        if cfg!(unix) {
            assert_eq!(
                std::fs::read(jre.join("legal/jdk.net/LICENSE")).unwrap(),
                b"GPL"
            );
        }
        assert!(!dir.join("jre.unpacking").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The real archive (STARLING_TEST_JRE: a downloaded jre25-android-arm64.tar.xz), timed.
    #[test]
    #[ignore = "needs the downloaded runtime archive"]
    fn the_android_runtime_unpacks() {
        let archive = PathBuf::from(std::env::var("STARLING_TEST_JRE").unwrap());
        let dir = temp("real");
        let started = std::time::Instant::now();
        unpack(&archive, &dir.join("jre")).unwrap();
        eprintln!("unpacked in {:?}", started.elapsed());
        for file in [
            "lib/libjli.so",
            "lib/server/libjvm.so",
            "lib/modules",
            "bin/java",
        ] {
            assert!(dir.join("jre").join(file).is_file(), "{file}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn entries_outside_the_folder_are_refused() {
        let dir = temp("escape");
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(1);
        header.set_mode(0o644);
        // append_data refuses `..`; the raw name goes through the header.
        header.as_gnu_mut().unwrap().name[..9].copy_from_slice(b"../escape");
        header.set_cksum();
        builder.append(&header, &b"x"[..]).unwrap();
        let tar = builder.into_inner().unwrap();
        let mut xz = Vec::new();
        lzma_rs::xz_compress(&mut &tar[..], &mut xz).unwrap();
        let archive = dir.join("bad.tar.xz");
        std::fs::write(&archive, xz).unwrap();
        let err = unpack(&archive, &dir.join("jre")).unwrap_err();
        assert!(err.contains("names"), "{err}");
        assert!(!dir.join("escape").exists() && !dir.join("jre").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
