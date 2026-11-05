#!/usr/bin/env python3
"""Prepare process-local Ampere twins from the user's own DLSS-G library.

Twin selection/shared-memory adjustment adapted from dlssg-open (MIT),
Copyright (c) 2026 JR Lanteigne; see LICENSE.dlssg-open.
This prototype selects only strict matches, rather than guessing incompatible layouts.
"""
from pathlib import Path
import hashlib
import json
import os
import struct


def elf(data):
    if len(data) < 64 or data[:5] != b'\x7fELF\x02' or struct.unpack_from('<H', data, 18)[0] != 190:
        return None
    phoff, shoff = struct.unpack_from('<QQ', data, 32)
    phsize, phnum, shsize, shnum, strings = struct.unpack_from('<HHHHH', data, 54)
    if shsize < 64 or not shnum or strings >= shnum or shoff+shsize*shnum > len(data):
        return None
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, shoff+i*shsize) for i in range(shnum)]
    string_section = sections[strings]
    names = data[string_section[4]:string_section[4]+string_section[5]]
    named = {}
    end = max(shoff+shsize*shnum, phoff+phsize*phnum)
    for i, section in enumerate(sections):
        name = names[section[0]:].split(b'\0', 1)[0].decode(errors='replace')
        named[name] = (i, section)
        if section[1] != 8:
            end = max(end, section[4]+section[5])
    if end > len(data): return None
    texts = [(name, section) for name, (_, section) in named.items() if name.startswith('.text.')]
    if not texts: return None
    kernel, section = texts[0]
    c2 = {name: data[s[4]:s[4]+s[5]] for name, (_, s) in named.items() if name.startswith('.nv.constant2')}
    c0 = {name: s[5] for name, (_, s) in named.items() if name.startswith('.nv.constant0')}
    shared = [(i, s[5]) for name, (i, s) in named.items() if name.startswith('.nv.shared.')]
    rw = []
    for i in range(phnum):
        kind, flags = struct.unpack_from('<II', data, phoff+i*phsize)
        if kind == 1 and flags & 2 and not flags & 1: rw.append(i)
    return dict(data=data[:end], sm=struct.unpack_from('<I', data, 48)[0]&255,
                kernel=kernel, code=data[section[4]:section[4]+section[5]], c2=c2, c0=c0,
                shared=shared, rw=rw, phoff=phoff, phsize=phsize, shoff=shoff, shsize=shsize)


library = Path(os.environ["HORIZON_DLSSG_RUNTIME"])
data = library.read_bytes()
twins = []
position = 0
while (position := data.find(b'\x7fELF', position)) >= 0:
    candidate = elf(data[position:])
    if candidate and candidate['sm'] == 86: twins.append(candidate)
    position += 4
print('Embedded sm_86 kernels:', len(twins))
out = Path('replacements'); out.mkdir(exist_ok=True)
manifest = []
for path in sorted(Path('logs').glob('module-*.bin')):
    incoming = elf(path.read_bytes())
    if not incoming or incoming['sm'] <= 86: continue
    matches = [t for t in twins if t['kernel'] == incoming['kernel']
               and len(t['code']) == len(incoming['code']) and t['c2'] == incoming['c2'] and t['c0'] == incoming['c0']]
    if not matches: raise SystemExit('No strict Ampere match: '+str(path)+' '+incoming['kernel'])
    def score(t):
        a, b = incoming['code'], t['code']
        return sum(a[i:i+16] == b[i:i+16] for i in range(0, len(a), 16))
    chosen = max(matches, key=score)
    replacement = bytearray(chosen['data'])
    shared = incoming['shared'][0][1] if incoming['shared'] else 0
    if shared:
        if len(chosen['shared']) != 1: raise SystemExit('Ambiguous shared memory layout')
        struct.pack_into('<Q', replacement, chosen['shoff']+chosen['shared'][0][0]*chosen['shsize']+32, shared)
        for i in chosen['rw']:
            struct.pack_into('<Q', replacement, chosen['phoff']+i*chosen['phsize']+40, shared)
    (out/path.name).write_bytes(replacement)
    (out/('expected-'+path.name)).write_bytes(path.read_bytes())
    manifest.append(dict(module=path.name, kernel=incoming['kernel'], shared=shared,
                         input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         output_sha256=hashlib.sha256(replacement).hexdigest()))
    print(path.name, incoming['kernel'], 'strict candidates', len(matches))
(out/'manifest.json').write_text(json.dumps(manifest, indent=2))
print('Prepared', len(manifest), 'strict replacements')
