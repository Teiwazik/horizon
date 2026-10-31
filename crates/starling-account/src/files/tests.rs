//! Downloads from a local mock of piston-meta and the resources server, with made-up files
//! (no game data): the layout, the hash checks and resuming.
use super::*;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::sync::Arc;

/// A one-thread HTTP/1.1 server answering GETs from `files` (path → body); it counts requests,
/// and closes the next `drops` connections unanswered, as a network that drops.
fn serve(
    files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    hits: Arc<AtomicU64>,
    drops: Arc<AtomicU64>,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            // One thread serves: nothing else changes `drops` between these two.
            if drops.load(Ordering::Relaxed) > 0 {
                drops.fetch_sub(1, Ordering::Relaxed);
                continue;
            }
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() {
                continue;
            }
            let path = line.split_whitespace().nth(1).unwrap_or("").to_owned();
            loop {
                let mut header = String::new();
                if reader.read_line(&mut header).is_err() || header.trim_end().is_empty() {
                    break;
                }
            }
            hits.fetch_add(1, Ordering::Relaxed);
            let body = files.lock().unwrap().get(&path).cloned();
            let (status, body) = match body {
                Some(b) => (200, b),
                None => (404, Vec::new()),
            };
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(&body);
        }
    });
    base
}

struct Fake {
    base: String,
    files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    hits: Arc<AtomicU64>,
    drops: Arc<AtomicU64>,
    pins: Pins,
}

/// A made-up version "T": its JSON, two jars, a library for Linux and one for Windows, and an
/// asset index of two objects.
fn fake() -> Fake {
    let files = Arc::new(Mutex::new(HashMap::new()));
    let hits = Arc::new(AtomicU64::new(0));
    let drops = Arc::new(AtomicU64::new(0));
    let base = serve(files.clone(), hits.clone(), drops.clone());
    let put = |path: &str, body: Vec<u8>| -> String {
        let sha1 = sha1_hex(&body);
        files.lock().unwrap().insert(path.to_owned(), body);
        sha1
    };
    let client = put("/client.jar", b"made-up client jar".to_vec());
    let server = put("/server.jar", b"made-up server jar".to_vec());
    let lib = put("/libs/a.jar", b"library a".to_vec());
    let windows = put("/libs/w.jar", b"windows only".to_vec());
    let (one, two) = (b"object one".to_vec(), b"object two".to_vec());
    let (h1, h2) = (sha1_hex(&one), sha1_hex(&two));
    put(&format!("/res/{}/{h1}", &h1[..2]), one);
    put(&format!("/res/{}/{h2}", &h2[..2]), two);
    let index = serde_json::json!({"objects": {
        "minecraft/lang/x.json": {"hash": h1, "size": 10},
        "minecraft/sounds/y.ogg": {"hash": h2, "size": 10},
        "bad": {"hash": "../../etc", "size": 1},
    }})
    .to_string();
    let index_sha1 = put("/index.json", index.into_bytes());
    let version = serde_json::json!({
        "id": "T",
        "downloads": {
            "client": {"url": format!("{base}/client.jar"), "sha1": client, "size": 18},
            "server": {"url": format!("{base}/server.jar"), "sha1": server, "size": 18},
        },
        "assetIndex": {"id": "9", "url": format!("{base}/index.json"), "sha1": index_sha1},
        "libraries": [
            {"name": "a:a:1", "downloads": {"artifact": {"path": "a/a/1/a.jar", "url": format!("{base}/libs/a.jar"), "sha1": lib, "size": 9}}},
            {"name": "w:w:1", "rules": [{"action": "allow", "os": {"name": "windows"}}],
             "downloads": {"artifact": {"path": "w/w/1/w.jar", "url": format!("{base}/libs/w.jar"), "sha1": windows, "size": 12}}},
        ],
    })
    .to_string();
    let manifest_sha1 = sha1_hex(version.as_bytes());
    put(
        &format!("/v1/packages/{manifest_sha1}/T.json"),
        version.into_bytes(),
    );
    Fake {
        pins: Pins {
            version: "T".into(),
            manifests: vec![manifest_sha1],
            client_sha1: client,
            server_sha1: server,
        },
        base,
        files,
        hits,
        drops,
    }
}

#[test]
fn a_network_that_drops_for_a_moment_is_tried_again() {
    let f = fake();
    let root = std::env::temp_dir().join(format!("starling-files-drop-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    // Two connections closed unanswered: the file they were for comes on the third try.
    f.drops.store(2, Ordering::Relaxed);
    let got = download(&sources(&f), &f.pins, &root, ALL, &|_| {}, &|| false).unwrap();
    assert_eq!(f.drops.load(Ordering::Relaxed), 0);
    assert_eq!(
        std::fs::read(&got.client_jar).unwrap(),
        b"made-up client jar"
    );
    let _ = std::fs::remove_dir_all(&root);
}

fn sources(f: &Fake) -> Sources {
    Sources {
        meta: f.base.clone(),
        resources: format!("{}/res", f.base),
    }
}

const ALL: Wanted = Wanted {
    server: true,
    libraries: true,
    assets: true,
};

#[test]
fn the_official_layout_is_checked_and_resumable() {
    let f = fake();
    let root = std::env::temp_dir().join(format!("starling-files-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let last = Mutex::new(Progress::default());
    let got = download(
        &sources(&f),
        &f.pins,
        &root,
        ALL,
        &|p| *last.lock().unwrap() = p,
        &|| false,
    )
    .unwrap();
    assert_eq!(got.client_jar, root.join("versions/T/T.jar"));
    assert_eq!(
        std::fs::read(&got.client_jar).unwrap(),
        b"made-up client jar"
    );
    assert_eq!(
        std::fs::read(got.server_jar.unwrap()).unwrap(),
        b"made-up server jar"
    );
    assert!(root.join("versions/T/T.json").is_file());
    assert!(root.join("libraries/a/a/1/a.jar").is_file());
    // Another OS's library is left out; a hash that is not one is never a path.
    assert!(!root.join("libraries/w").exists());
    assert!(root.join("assets/indexes/9.json").is_file());
    let h1 = sha1_hex(b"object one");
    assert!(
        root.join("assets/objects")
            .join(&h1[..2])
            .join(&h1)
            .is_file()
    );
    let p = *last.lock().unwrap();
    assert_eq!((p.files, p.files_total), (5, 5));
    // Again: everything is there with its hash, so only nothing is fetched.
    let before = f.hits.load(Ordering::Relaxed);
    download(&sources(&f), &f.pins, &root, ALL, &|_| {}, &|| false).unwrap();
    assert_eq!(f.hits.load(Ordering::Relaxed), before);
    // A damaged file is fetched again; no `.part` file stays behind.
    std::fs::write(root.join("libraries/a/a/1/a.jar"), b"damaged").unwrap();
    download(&sources(&f), &f.pins, &root, ALL, &|_| {}, &|| false).unwrap();
    assert_eq!(
        std::fs::read(root.join("libraries/a/a/1/a.jar")).unwrap(),
        b"library a"
    );
    assert!(!root.join("libraries/a/a/1/a.jar.part").exists());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_file_that_is_not_the_listed_one_is_refused() {
    let f = fake();
    let root = std::env::temp_dir().join(format!("starling-files-bad-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    f.files
        .lock()
        .unwrap()
        .insert("/libs/a.jar".into(), b"something else".to_vec());
    let err = download(&sources(&f), &f.pins, &root, ALL, &|_| {}, &|| false).unwrap_err();
    assert_eq!(
        err,
        "a.jar is not the file Mojang lists (its hash differs)."
    );
    assert!(!root.join("libraries/a/a/1/a.jar").exists());
    // A version JSON Starling does not pin is never used.
    let mut other = f.pins.clone();
    other.manifests = vec!["0".repeat(40)];
    let root2 = root.join("other");
    assert!(
        download(&sources(&f), &other, &root2, ALL, &|_| {}, &|| false)
            .unwrap_err()
            .contains("refused")
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn starlings_pins_are_the_research_lock() {
    let pins = Pins::starling();
    assert_eq!(pins.version, "26.3");
    // The pinned version JSON, then every republished one the lock accepts (Horizon adds them).
    let lock: serde_json::Value =
        serde_json::from_str(include_str!("../../../../research-lock.json")).unwrap();
    let accepted = lock["minecraft"]["accepted_manifest_sha1"]
        .as_array()
        .map_or(0, Vec::len);
    assert_eq!(pins.manifests.len(), 1 + accepted);
    assert!(pins.client_sha1.len() == 40 && pins.server_sha1.len() == 40);
}
