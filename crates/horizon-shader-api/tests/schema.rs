//! Schema 1 of `horizon-shaders.toml` and the `.hshader` package.
use horizon_shader_api::{MANIFEST, Manifest, OptionKind, Pack, PageItem};
use std::{collections::BTreeMap, path::Path};

fn sample_dir() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/samples/schema-1"
    ))
}

fn sample_text() -> String {
    std::fs::read_to_string(sample_dir().join(MANIFEST)).unwrap()
}

fn refused(text: &str) -> String {
    Manifest::parse(text).expect_err("should be refused")
}

#[test]
fn reads_every_key_of_schema_1() {
    let m = Manifest::parse(&sample_text()).unwrap();
    assert_eq!(
        (
            m.id.as_str(),
            m.name.as_str(),
            m.version.as_str(),
            m.api.as_str()
        ),
        ("sample_pack", "Sample Pack", "1.0.0", "0.5")
    );
    assert_eq!(m.authors, ["Horizon"]);
    assert_eq!(m.license.as_deref(), Some("MIT"));
    assert_eq!(m.sky.as_ref().unwrap().file, "sky.wgsl");
    assert_eq!(
        m.material.as_ref().unwrap().name.as_deref(),
        Some("Terrain")
    );
    let post: Vec<_> = m.post.iter().map(|s| s.file.as_str()).collect();
    assert_eq!(post, ["post/grade.wgsl", "post/vignette.wgsl"]);
    assert_eq!(
        m.files(),
        [
            "common.wgsl",
            "sky.wgsl",
            "material.wgsl",
            "post/grade.wgsl",
            "post/vignette.wgsl"
        ]
    );
    let kinds: Vec<_> = m.options.iter().map(|o| (o.id.as_str(), &o.kind)).collect();
    assert_eq!(
        kinds,
        [
            (
                "saturation",
                &OptionKind::Float {
                    default: 1.1,
                    min: 0.0,
                    max: 2.0,
                    step: 0.05
                }
            ),
            ("vignette", &OptionKind::Bool { default: true }),
            (
                "samples",
                &OptionKind::Int {
                    default: 4,
                    min: 1,
                    max: 16
                }
            ),
            (
                "tonemap",
                &OptionKind::Choice {
                    values: vec!["None".into(), "Soft".into(), "Filmic".into()],
                    default: 1
                }
            ),
        ]
    );
    let types: Vec<_> = m.options.iter().map(|o| o.kind.wgsl_type()).collect();
    assert_eq!(types, ["f32", "bool", "i32", "u32"]);
    let constants: Vec<_> = m
        .options
        .iter()
        .map(|o| o.kind.default_constant())
        .collect();
    assert_eq!(constants, [f64::from(1.1f32), 1.0, 4.0, 1.0]);
    assert_eq!(m.budget.gpu_ms, Some(1.5));
}

#[test]
fn the_smallest_manifest() {
    let m = Manifest::parse(
        "schema = 1\nid = \"p\"\nname = \"P\"\nversion = \"1\"\napi = \"0.5\"\n[[post]]\nfile = \"a.wgsl\"\n",
    )
    .unwrap();
    assert!(m.sky.is_none() && m.options.is_empty() && m.budget.gpu_ms.is_none());
}

#[test]
fn refuses_what_this_build_does_not_read() {
    let text = sample_text();
    let cases = [
        (text.replace("schema = 1", "schema = 2"), "schema 2"),
        (
            text.replace("api = \"0.5\"", "api = \"0.6\""),
            "shader API 0.6",
        ),
        (
            text.replace("\"sample_pack\"", "\"Sample\""),
            "is not [a-z0-9_]",
        ),
        (
            text.replace("license = \"MIT\"", "licence = \"MIT\""),
            "unknown field",
        ),
        (
            text.replace("gpu_ms = 1.5", "gpu_ms = 40.0"),
            "budget.gpu_ms",
        ),
        (
            text.replace("\"common.wgsl\"", "\"../common.wgsl\""),
            "not a plain path",
        ),
        (
            text.replace("\"sky.wgsl\"", "\"sky.glsl\""),
            "not a plain path to a .wgsl",
        ),
        (
            text.replace("id = \"samples\"", "id = \"vignette\""),
            "declared twice",
        ),
        (
            text.replace("id = \"samples\"", "id = \"horizon_samples\""),
            "WGSL identifier",
        ),
        (
            text.replace("default = \"Soft\"", "default = \"Hard\""),
            "one of the values",
        ),
        (
            text.replace("default = 1.1", "default = 3.0"),
            "default in [0, 2]",
        ),
        (
            text.replace("default = 4", "default = 4.5"),
            "not a 32-bit integer",
        ),
        (
            text.replace("type = \"bool\"", "type = \"color\""),
            "is not bool",
        ),
        (
            text.replace("default = true", "default = true\nmin = 0"),
            "doesn't apply",
        ),
    ];
    for (text, needle) in cases {
        let e = refused(&text);
        assert!(e.contains(needle), "{needle:?} not in {e:?}");
        assert!(e.starts_with(MANIFEST), "{e}");
    }
    let no_stage = "schema = 1\nid = \"p\"\nname = \"P\"\nversion = \"1\"\napi = \"0.5\"\n";
    assert!(refused(no_stage).contains("no stage"));
    let many = (0..13).map(|i| format!("[[post]]\nfile = \"p{i}.wgsl\"\n"));
    assert!(refused(&format!("{no_stage}{}", many.collect::<String>())).contains("13 post passes"));
}

#[test]
fn reads_the_sample_folder_and_its_package_alike() {
    let folder = Pack::read(sample_dir()).unwrap();
    assert_eq!(folder.files.len(), 7);
    assert!(
        folder
            .source("post/grade.wgsl")
            .unwrap()
            .contains("fn post")
    );
    let bytes = folder.package().unwrap();
    assert_eq!(
        bytes,
        folder.package().unwrap(),
        "packaging is reproducible"
    );
    let dir = std::env::temp_dir().join(format!("horizon-shader-api-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sample_pack.hshader");
    std::fs::write(&path, &bytes).unwrap();
    let packaged = Pack::read(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(packaged.files, folder.files);
    assert_eq!(packaged.manifest, folder.manifest);
}

#[test]
fn refuses_files_a_pack_does_not_use() {
    let base = || -> BTreeMap<String, Vec<u8>> { Pack::read(sample_dir()).unwrap().files };
    let mut files = base();
    files.insert("old.wgsl".into(), b"fn f() {}".to_vec());
    assert!(
        Pack::from_files(files)
            .unwrap_err()
            .contains("old.wgsl is not used")
    );
    let mut files = base();
    files.insert("run.sh".into(), Vec::new());
    assert!(
        Pack::from_files(files)
            .unwrap_err()
            .contains("run.sh: a pack holds only")
    );
    let mut files = base();
    files.remove("sky.wgsl");
    assert!(
        Pack::from_files(files)
            .unwrap_err()
            .contains("sky.wgsl: named in")
    );
    let mut files = base();
    files.insert("sky.wgsl".into(), vec![0xff, 0xfe]);
    assert!(Pack::from_files(files).unwrap_err().contains("not UTF-8"));
    let mut files = base();
    files.insert("post/grade.wgsl".into(), vec![b' '; 300 << 10]);
    assert!(
        Pack::from_files(files)
            .unwrap_err()
            .contains("over 256 KiB")
    );
    let mut files = base();
    files.remove(MANIFEST);
    assert!(
        Pack::from_files(files)
            .unwrap_err()
            .contains("missing at the pack's root")
    );
}

#[test]
fn reads_materials_textures_the_vertex_stage_and_the_sky_keys() {
    let m = Manifest::parse(
        "schema = 1\nid = \"p\"\nname = \"P\"\nversion = \"1\"\napi = \"0.5\"\n\
         [sky]\nfile = \"sky.wgsl\"\nclouds = false\nsun = false\nstars = false\n\
         [material]\nfile = \"m.wgsl\"\nfog = false\n\
         [vertex]\nfile = \"v.wgsl\"\n\
         [[textures]]\nid = \"noise\"\nfile = \"noise.png\"\nfilter = \"nearest\"\nwrap = \"clamp\"\n\
         [[materials]]\nid = 2\nsprites = [\"block/*_leaves\", \"block/vine\"]\n\
         [[materials]]\nid = 3\nsprites = [\"mod:block/glow*\", \"block/*\"]\n",
    )
    .unwrap();
    assert_eq!(m.sky.as_ref().unwrap().clouds, Some(false));
    assert_eq!(m.sky.as_ref().unwrap().sun, Some(false));
    assert_eq!(m.sky.as_ref().unwrap().stars, Some(false));
    assert_eq!(m.material.as_ref().unwrap().fog, Some(false));
    assert_eq!(m.vertex.as_ref().unwrap().file, "v.wgsl");
    let t = &m.textures[0];
    assert_eq!(
        (t.id.as_str(), t.file.as_str(), t.linear, t.repeat),
        ("noise", "noise.png", false, false)
    );
    // The first entry that matches wins; a pattern without a namespace means any.
    assert_eq!(m.material_of("block/oak_leaves"), 2);
    assert_eq!(m.material_of("minecraft:block/vine"), 2);
    assert_eq!(m.material_of("other:block/oak_leaves"), 2);
    assert_eq!(m.material_of("mod:block/glowing"), 3);
    assert_eq!(m.material_of("block/stone"), 3);
    assert_eq!(m.material_of("item/stick"), 0);
}

#[test]
fn refuses_the_new_keys_where_they_do_not_belong() {
    let head = "schema = 1\nid = \"p\"\nname = \"P\"\nversion = \"1\"\napi = \"0.5\"\n";
    let cases = [
        (
            "[[post]]\nfile = \"a.wgsl\"\nclouds = false\n",
            "`clouds` is a key of [sky] only",
        ),
        (
            "[[post]]\nfile = \"a.wgsl\"\nsun = false\n",
            "`sun` is a key of [sky] only",
        ),
        (
            "[[post]]\nfile = \"a.wgsl\"\nstars = false\n",
            "`stars` is a key of [sky] only",
        ),
        (
            "[[post]]\nfile = \"a.wgsl\"\nfog = false\n",
            "`fog` is a key of [material] only",
        ),
        (
            "[[post]]\nfile = \"a.wgsl\"\n[vertex]\nfile = \"v.wgsl\"\n",
            "[vertex] needs a [material]",
        ),
        (
            "[[post]]\nfile = \"a.wgsl\"\n[[materials]]\nid = 0\nsprites = []\n",
            "material id 0",
        ),
        (
            "[[post]]\nfile = \"a.wgsl\"\n[[textures]]\nid = \"t\"\nfile = \"t.jpg\"\n",
            ".png file",
        ),
    ];
    for (body, needle) in cases {
        let e = refused(&format!("{head}{body}"));
        assert!(e.contains(needle), "{needle:?} not in {e:?}");
    }
}

#[test]
fn reads_settings_pages_and_refuses_broken_links() {
    let pages = "[[pages]]\nid = \"main\"\nname = \"Sample\"\nitems = [\"saturation\", \"\", \"page:more\"]\n\
                 [[pages]]\nid = \"more\"\nname = \"More\"\ndescription = \"The rest.\"\ncolumns = 1\n\
                 items = [\"tonemap\", \"samples\", \"vignette\"]\n";
    let m = Manifest::parse(&format!("{}{pages}", sample_text())).unwrap();
    assert_eq!(m.pages.len(), 2);
    assert_eq!(
        m.pages[0].items,
        [PageItem::Option(0), PageItem::Empty, PageItem::Page(1)]
    );
    assert_eq!(
        (m.pages[1].columns, m.pages[1].description.as_str()),
        (1, "The rest.")
    );
    assert_eq!(
        m.pages[1].items,
        [PageItem::Option(3), PageItem::Option(2), PageItem::Option(1)]
    );
    // Without pages, the settings show every option on one.
    assert!(Manifest::parse(&sample_text()).unwrap().pages.is_empty());
    let cases = [
        (pages.replace("\"page:more\"", "\"page:less\""), "no page \"less\""),
        (pages.replace("\"page:more\"", "\"page:main\""), "links to itself"),
        (pages.replace("\"tonemap\"", "\"tone\""), "no option \"tone\""),
        (pages.replace("\"samples\",", "\"tonemap\","), "on the page twice"),
        (pages.replace("columns = 1", "columns = 3"), "1 or 2"),
        (pages.replace("id = \"more\"", "id = \"main\""), "declared twice"),
        (pages.replace(", \"vignette\"]", "]"), "option \"vignette\" is on no page"),
        (pages.replace("\"page:more\"", "\"\""), "page \"more\" can't be reached"),
    ];
    for (pages, needle) in cases {
        let e = refused(&format!("{}{pages}", sample_text()));
        assert!(e.contains(needle), "{needle:?} not in {e:?}");
    }
}

#[test]
fn reads_solid_shadow_materials_and_refuses_unknown_ids() {
    let head = "schema = 1\nid = \"p\"\nname = \"P\"\nversion = \"1\"\napi = \"0.5\"\n\
                [[post]]\nfile = \"a.wgsl\"\n[[materials]]\nid = 2\nsprites = [\"block/*_leaves\"]\n";
    let m = Manifest::parse(&format!("{head}[shadow]\nsolid_materials = [2]\n")).unwrap();
    assert_eq!(m.shadow.unwrap().solid_materials, [2]);
    let e = refused(&format!("{head}[shadow]\nsolid_materials = [3]\n"));
    assert!(e.contains("no [[materials]] entry has id 3"), "{e}");
}

#[test]
fn reads_reflection_and_its_option() {
    let head = "schema = 1
id = \"t\"
name = \"T\"
version = \"1\"
api = \"0.5\"
";
    let material = "[material]
file = \"m.wgsl\"
";
    let option = "[[options]]
id = \"mirror\"
name = \"Mirror\"
type = \"float\"
default = 1.0
min = 0.0
max = 1.0
";
    let m = Manifest::parse(&format!(
        "{head}{material}{option}[reflection]
option = \"mirror\"
"
    ))
    .unwrap();
    assert_eq!(m.reflection.unwrap().option.as_deref(), Some("mirror"));
    assert!(Manifest::parse(&format!("{head}{material}")).unwrap().reflection.is_none());
    assert!(
        refused(&format!("{head}[[post]]
file = \"p.wgsl\"
[reflection]
"))
            .contains("needs [material]")
    );
    assert!(
        refused(&format!("{head}{material}[reflection]
option = \"nope\"
"))
            .contains("is no option")
    );
}

#[test]
fn reads_post_outputs_and_their_scales() {
    let head = "schema = 1\nid = \"t\"\nname = \"T\"\nversion = \"1\"\napi = \"0.5\"\n";
    let pass = |file: &str, extra: &str| format!("[[post]]\nfile = \"{file}\"\n{extra}");
    let m = Manifest::parse(&format!(
        "{head}{}{}{}{}",
        pass("a.wgsl", "output = \"glow\"\nscale = 0.5\n"),
        pass("b.wgsl", "output = \"soft\"\nscale = 0.25\n"),
        pass("c.wgsl", "output = \"glow\"\nscale = 0.5\n"),
        pass("d.wgsl", "")
    ))
    .unwrap();
    assert_eq!(m.post_targets(), [("glow", 0.5), ("soft", 0.25)]);
    let one = |extra: &str| format!("{head}{}{}", pass("a.wgsl", extra), pass("z.wgsl", ""));
    assert!(refused(&one("scale = 0.5\n")).contains("writes an `output`"));
    assert!(refused(&one("scale = 0.3\noutput = \"x\"\n")).contains("is not one of"));
    assert!(refused(&one("output = \"previous\"\n")).contains("not a free WGSL name"));
    assert!(refused(&format!("{head}{}", pass("a.wgsl", "output = \"x\"\n"))).contains("last"));
    assert!(
        refused(&format!("{head}[sky]\nfile = \"s.wgsl\"\nscale = 0.5\n"))
            .contains("[[post]] only")
    );
}
