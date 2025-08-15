#!/usr/bin/env python3
"""
cpp_preprocessor_debug_full.py (patched)

Same as previous verbose preprocessor, but with namespace-fallback:
- If namespace brace-matching fails, still scan the namespace substring for class blocks,
  process them and wrap their collected out-of-line defs in `namespace <name> { ... }`.
- Extensive DBG/INFO output to trace behavior.
"""

from typing import Dict, Callable, List, Tuple
import os, sys, re, time, importlib.util, inspect, traceback
from collections import defaultdict

# ---------------------------
# Config
# ---------------------------
try:
    import config  # type: ignore
except Exception as e:
    raise RuntimeError("Please create config.py next to this script defining file_extension") from e

if not hasattr(config, "file_extension"):
    raise RuntimeError("config.py must define 'file_extension' (e.g. file_extension = 'cxx')")

FILE_EXT = config.file_extension.strip().lstrip(".")
ATTR_DIR = getattr(config, "attributes_dir", ".")
TARGET_DIR = getattr(config, "target_dir", ".")
RECURSIVE = getattr(config, "recursive", True)

META_FILE = os.path.join(TARGET_DIR, "edit.meta")
BUILD_HEADER_DIR = os.path.join(".", "build", "header")
BUILD_SRC_DIR = os.path.join(".", "build", "src")

# ---------------------------
# Logging helpers (very verbose)
# ---------------------------
def info(msg: str):
    print("[INFO]", msg)

def dbg(msg: str):
    print("[DBG]", msg)

def warn(msg: str):
    print("[WARN]", msg, file=sys.stderr)

def err(msg: str):
    print("[ERROR]", msg, file=sys.stderr)

# ---------------------------
# edit.meta helpers
# ---------------------------
def load_edit_meta(path: str) -> Dict[str, float]:
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or "=" not in s: continue
            k, v = s.split("=", 1)
            try:
                d[k.strip()] = float(v.strip())
            except Exception:
                continue
    return d

def write_edit_meta(path: str, d: Dict[str, float]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for k, v in sorted(d.items()):
            f.write(f"{k}={v}\n")
    os.replace(tmp, path)

# ---------------------------
# dynamic loader for attribute functions
# ---------------------------
def load_functions_from_dir(directory: str) -> Dict[str, Callable[[str], str]]:
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        info(f"attributes dir '{directory}' not found -> no attributes loaded")
        return {}
    deps = os.path.join(directory, "dependencies")
    if os.path.isdir(deps) and deps not in sys.path:
        sys.path.insert(0, deps)
        dbg(f"Inserted dependencies path into sys.path: {deps}")

    script_name = os.path.basename(__file__)
    pyfiles = sorted([f for f in os.listdir(directory)
                      if f.endswith(".py") and f not in ("__init__.py", "config.py", script_name)])

    info(f"Attribute Python files: {pyfiles}")
    name_to_funcs = defaultdict(list)
    for py in pyfiles:
        full = os.path.join(directory, py)
        spec_name = f"__attr_{os.path.splitext(py)[0]}"
        try:
            spec = importlib.util.spec_from_file_location(spec_name, full)
            if spec is None:
                warn(f"spec None for {full}")
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec_name] = module
            loader = spec.loader
            if loader is None:
                warn(f"no loader for {full}")
                continue
            loader.exec_module(module)
            funcs_count = 0
            for nm, fn in inspect.getmembers(module, inspect.isfunction):
                if getattr(fn, "__module__", None) != module.__name__:
                    continue
                name_to_funcs[nm].append(fn)
                funcs_count += 1
            info(f"Imported module {py} with {funcs_count} local functions")
        except Exception as e:
            warn(f"Failed to import {full}: {e}")
            warn(traceback.format_exc())
            continue

    result = {}
    for name, funcs in name_to_funcs.items():
        if len(funcs) == 1:
            result[name] = funcs[0]
        else:
            def make_chain(fs):
                def chain(s: str):
                    out = s
                    for f in fs:
                        out = f(out)
                    return out
                return chain
            result[name] = make_chain(funcs)
            info(f"Chained {len(funcs)} functions into attribute '{name}'")
    return dict(result)

# ---------------------------
# Brace-aware helpers
# ---------------------------
def find_matching_brace(text: str, open_pos: int) -> int:
    n = len(text)
    if open_pos < 0 or open_pos >= n or text[open_pos] != "{":
        return -1
    depth = 1
    i = open_pos + 1
    while i < n:
        if text.startswith("//", i):
            nl = text.find("\n", i)
            if nl == -1: return -1
            i = nl + 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i+2)
            if end == -1: return -1
            i = end + 2
            continue
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1

# ---------------------------
# Attribute application
# ---------------------------
ATTR_LINE_RE = re.compile(r'^[ \t]*//\s*@([A-Za-z_]\w*)\s*$', re.MULTILINE)

def find_next_nonblank(text: str, start: int) -> int:
    n = len(text)
    i = start
    while i < n:
        nl = text.find("\n", i)
        if nl == -1:
            return i if text[i:].strip() else n
        line = text[i:nl]
        if line.strip() == "":
            i = nl + 1
            continue
        return i
    return n

def extract_block_after_attr(text: str, after_idx: int) -> Tuple[int,int]:
    n = len(text)
    i = find_next_nonblank(text, after_idx)
    if i >= n:
        return (after_idx, after_idx)
    brace_pos = text.find("{", i)
    if brace_pos == -1:
        nl = text.find("\n", i)
        if nl == -1: return (i, n-1)
        return (i, nl)
    start = i
    prev_nl = text.rfind("\n", 0, start)
    if prev_nl != -1:
        prev_line_start = text.rfind("\n", 0, prev_nl) + 1
        prev_line = text[prev_line_start:prev_nl].strip()
        if prev_line.startswith("template"):
            start = prev_line_start
    close = find_matching_brace(text, brace_pos)
    if close == -1:
        raise RuntimeError("unmatched brace after attribute")
    j = close + 1
    while j < n and text[j].isspace(): j += 1
    if j < n and text[j] == ";":
        close = j
    return (start, close)

def apply_attributes_to_text(text: str, funcs: Dict[str, Callable[[str], str]]) -> str:
    out_parts = []
    last = 0
    for m in ATTR_LINE_RE.finditer(text):
        attr = m.group(1)
        s = m.start(); e = m.end()
        out_parts.append(text[last:s])
        try:
            blk_s, blk_e = extract_block_after_attr(text, e+1)
        except Exception as ex:
            raise RuntimeError(f"Failed to extract block after @{attr} at pos {e}: {ex}")
        block_text = text[blk_s:blk_e+1]
        info(f"Attribute @{attr} at bytes [{s},{e}) applies to block bytes [{blk_s},{blk_e}] (len={len(block_text)})")
        if attr not in funcs:
            raise KeyError(f"Attribute function '{attr}' not found in loaded functions")
        try:
            replaced = funcs[attr](block_text)
        except Exception as ex:
            tb = traceback.format_exc()
            raise RuntimeError(f"Attribute function '{attr}' raised: {ex}\n{tb}")
        out_parts.append(replaced)
        last = blk_e + 1
    out_parts.append(text[last:])
    return "".join(out_parts)

# ---------------------------
# include rewriting
# ---------------------------
def rewrite_includes(text: str, ext: str) -> str:
    pat = re.compile(r'#include\s*"([^"]+)\.' + re.escape(ext) + r'"')
    new = pat.sub(r'#include "\1.hpp"', text)
    info(f"Rewrote includes for extension '.{ext}'")
    return new

# ---------------------------
# Helpers for replacements
# ---------------------------
def apply_replacements_to_text(text: str, replacements: List[Tuple[int,int,str]]) -> str:
    if not replacements:
        return text
    res = text
    reps = sorted(replacements, key=lambda x: x[0], reverse=True)
    for s,e,new in reps:
        dbg(f"Applying replacement span [{s},{e}] -> new_len={len(new)}")
        if s < 0 or e >= len(res) or s > e:
            warn(f"Invalid replacement span [{s},{e}] for current text length {len(res)}; skipping")
            continue
        res = res[:s] + new + res[e+1:]
    return res

def get_span_text(text: str, span: Tuple[int,int]) -> str:
    s,e = span
    return text[s:e+1]

# ---------------------------
# Block extraction helpers
# ---------------------------
INCLUDE_RE = re.compile(r'^\s*#include\s+["<].+[">]')
NAMESPACE_RE = re.compile(r'\bnamespace\b\s*([A-Za-z_]\w*)?\s*\{')
CLASS_RE = re.compile(r'\b(class|struct)\b\s+([A-Za-z_]\w*)')
METHOD_CANDIDATE_RE = re.compile(r'([^\n;{}]+?\([^\)]*\))\s*(?:\b(const|noexcept|override|final)\b|[^;{]*)\s*\{', re.DOTALL)

def extract_top_includes(text: str) -> Tuple[List[Tuple[int,int,str]], str]:
    lines = text.splitlines(keepends=True)
    includes = []
    i = 0
    for idx, ln in enumerate(lines):
        if INCLUDE_RE.match(ln):
            start = sum(len(l) for l in lines[:idx])
            end = start + len(ln) - 1
            includes.append((start, end, ln))
            i = idx + 1
        elif ln.strip() == "":
            i = idx + 1
            continue
        else:
            break
    remaining = "".join(lines[i:])
    return includes, remaining

def extract_blocks_positions(text: str, keyword_re: re.Pattern) -> List[Tuple[int,int]]:
    spans = []
    i = 0
    n = len(text)
    while i < n:
        m = keyword_re.search(text, i)
        if not m:
            break
        start = m.start()
        brace_pos = text.find("{", m.end())
        if brace_pos == -1:
            warn(f"No '{{' after keyword at pos {m.end()}, stopping search")
            break
        close = find_matching_brace(text, brace_pos)
        if close == -1:
            warn(f"Unmatched brace for block starting at {start}; stopping search")
            break
        spans.append((start, close))
        dbg(f"Found block span [{start},{close}] for keyword at {m.start()} (len={close-start+1})")
        i = close + 1
    return spans

# ---------------------------
# Process class block -> produce new class text and out-of-line defs (strings)
# ---------------------------
def process_class_block_for_out_of_line(class_text: str, namespace_prefix: str="") -> Tuple[str, List[str]]:
    dbg("Processing class block of length {}".format(len(class_text)))
    mname = CLASS_RE.search(class_text)
    if not mname:
        dbg("No class/struct name found in class block")
        return class_text, []
    class_name = mname.group(2)
    first_brace = class_text.find("{", mname.end())
    if first_brace == -1:
        dbg("No opening brace for class body")
        return class_text, []
    last_brace = find_matching_brace(class_text, first_brace)
    if last_brace == -1:
        dbg("No matching closing brace for class body")
        return class_text, []
    dbg(f"Class '{class_name}' body bytes: [{first_brace},{last_brace}] (len={last_brace-first_brace+1})")
    inner = class_text[first_brace+1:last_brace]
    out_inner_parts = []
    out_of_line_defs = []
    idx = 0
    n = len(inner)
    while idx < n:
        m = METHOD_CANDIDATE_RE.search(inner, idx)
        if not m:
            out_inner_parts.append(inner[idx:])
            break
        sig_start = m.start(1)
        brace_open_rel = inner.find("{", m.end(1)-1)
        if brace_open_rel == -1:
            out_inner_parts.append(inner[idx:m.end()])
            idx = m.end()
            continue
        abs_brace_open = first_brace + 1 + brace_open_rel
        abs_brace_close = find_matching_brace(class_text, abs_brace_open)
        if abs_brace_close == -1:
            warn("Could not match method body brace inside class; skipping extraction for this candidate")
            out_inner_parts.append(inner[idx:m.end()])
            idx = m.end()
            continue
        rel_close = abs_brace_close - (first_brace + 1)
        pre = inner[idx:sig_start]
        signature = inner[sig_start:brace_open_rel].strip()
        body_with_braces = inner[brace_open_rel:rel_close+1]
        before_paren = signature.split("(", 1)[0]
        tokens = re.findall(r'([A-Za-z_]\w*)', before_paren)
        if tokens:
            method_name = tokens[-1]
        else:
            mm = re.search(r'([A-Za-z_]\w*)\s*\($', signature)
            method_name = mm.group(1) if mm else None
        if not method_name:
            warn("No method name found; leaving in-class body unchanged for this candidate")
            out_inner_parts.append(inner[idx:rel_close+1])
            idx = rel_close + 1
            continue
        decl = signature.rstrip() + ";"
        last_pos = before_paren.rfind(method_name)
        if last_pos != -1:
            qualified_before = before_paren[:last_pos] + f"{class_name}::{method_name}" + before_paren[last_pos+len(method_name):]
            rest_after = ""
            if "(" in signature:
                rest_after = "(" + signature.split("(",1)[1]
            qualified_sig = qualified_before + rest_after
        else:
            if "(" in signature:
                qualified_sig = signature.replace(method_name + "(", f"{class_name}::{method_name}(", 1)
            else:
                qualified_sig = f"{class_name}::{signature}"
        out_def = qualified_sig + " " + body_with_braces
        dbg(f"Found method '{method_name}' in class '{class_name}': decl='{decl[:120]}' qualified_head='{qualified_sig[:120]}'")
        out_inner_parts.append(pre)
        out_inner_parts.append(decl)
        out_of_line_defs.append(out_def)
        idx = rel_close + 1
    new_inner = "".join(out_inner_parts)
    new_class_text = class_text[:first_brace+1] + new_inner + class_text[last_brace:]
    new_class_text = re.sub(r';\s*;', ';', new_class_text)
    return new_class_text, out_of_line_defs

# ---------------------------
# Splitting (verbose) with namespace fallback
# ---------------------------
def split_header_and_source_verbose(processed_text: str, basename: str) -> Tuple[str, str]:
    try:
        text = processed_text
        info(f"Starting split for '{basename}' (input len={len(text)})")

        includes_spans, _ = extract_top_includes(text) if 'extract_top_includes' in globals() else ([], "")
        # if function not present in scope (older copies) reimplement quick includes extraction:
        if not includes_spans:
            # simple top-include extraction
            lines = text.splitlines(keepends=True)
            includes_spans = []
            for idx, ln in enumerate(lines):
                if re.match(r'^\s*#include\s+["<].+[">]', ln):
                    start = sum(len(l) for l in lines[:idx])
                    end = start + len(ln) - 1
                    includes_spans.append((start, end, ln))
                else:
                    break

        dbg(f"Top includes count: {len(includes_spans)}")
        for s,e,line in includes_spans:
            dbg(f"  include span [{s},{e}] -> '{line.strip()}'")
        header_includes_text = "".join([line for (_,_,line) in includes_spans]) if includes_spans else ""
        last_include_off = includes_spans[-1][1]+1 if includes_spans else 0

        rest_text = text[last_include_off:]
        ns_spans_rel = extract_blocks_positions(rest_text, NAMESPACE_RE)
        dbg(f"Namespace spans found (relative to rest): {ns_spans_rel}")
        ns_spans_abs = [(last_include_off + s, last_include_off + e) for (s,e) in ns_spans_rel]
        dbg(f"Namespace spans absolute: {ns_spans_abs}")

        replacements: List[Tuple[int,int,str]] = []
        collected_out_of_line_defs: List[str] = []

        for (s_abs, e_abs) in ns_spans_abs:
            ns_block = text[s_abs:e_abs+1]
            dbg(f"Processing namespace block bytes [{s_abs},{e_abs}] len={len(ns_block)}")
            mnm = re.match(r'\s*namespace\s+([A-Za-z_]\w*)', ns_block)
            ns_name = mnm.group(1) if mnm else ""
            dbg(f"  namespace name detected: '{ns_name}'")
            inner_brace_pos = ns_block.find("{")
            if inner_brace_pos == -1:
                dbg("  namespace has no inner brace; skipping class extraction inside namespace")
                # still queue it unchanged
                replacements.append((s_abs, e_abs, ns_block))
                continue
            inner_close = find_matching_brace(ns_block, inner_brace_pos)
            if inner_close == -1:
                # Fallback: attempt to find class spans inside the ns_block text directly (best-effort)
                dbg("  could not find inner close for namespace; falling back to scanning namespace text for classes")
                class_spans_rel_in_ns = extract_blocks_positions(ns_block, CLASS_RE)
                dbg(f"  fallback found {len(class_spans_rel_in_ns)} class spans inside namespace text")
                if not class_spans_rel_in_ns:
                    replacements.append((s_abs, e_abs, ns_block))
                    continue
                inner_repls = []
                defs_for_ns = []
                for (cs, ce) in class_spans_rel_in_ns:
                    class_text = ns_block[cs:ce+1]
                    dbg(f"    class span in ns_block [{cs},{ce}] len={len(class_text)}")
                    new_class_text, defs = process_class_block_for_out_of_line(class_text, namespace_prefix="")
                    dbg(f"      produced new_class_text len={len(new_class_text)} and {len(defs)} defs")
                    inner_repls.append((cs, ce, new_class_text))
                    defs_for_ns.extend(defs)
                # apply inner replacements to ns_block
                new_inner = apply_replacements_to_text(ns_block, inner_repls)
                # queue namespace replacement
                replacements.append((s_abs, e_abs, new_inner))
                if defs_for_ns:
                    wrapped = f"namespace {ns_name} {{\n\n" + "\n\n".join(defs_for_ns) + "\n\n}\n"
                    dbg(f"  Wrapping {len(defs_for_ns)} defs into namespace '{ns_name}' (len={len(wrapped)})")
                    collected_out_of_line_defs.append(wrapped)
                continue

            # If we successfully found inner_close, process classes inside normally
            inner = ns_block[inner_brace_pos+1:inner_close]
            class_spans_rel = extract_blocks_positions(inner, CLASS_RE)
            dbg(f"  Found {len(class_spans_rel)} class spans inside namespace (rel to inner): {class_spans_rel}")
            if not class_spans_rel:
                replacements.append((s_abs, e_abs, ns_block))
                continue
            inner_repls = []
            defs_for_ns = []
            for (cs, ce) in class_spans_rel:
                class_text = inner[cs:ce+1]
                dbg(f"    class span in inner [{cs},{ce}] len={len(class_text)}")
                new_class_text, defs = process_class_block_for_out_of_line(class_text, namespace_prefix="")
                dbg(f"      produced new_class_text len={len(new_class_text)} and {len(defs)} defs")
                inner_repls.append((cs, ce, new_class_text))
                defs_for_ns.extend(defs)
            new_inner = apply_replacements_to_text(inner, inner_repls)
            new_ns_block = ns_block[:inner_brace_pos+1] + new_inner + ns_block[inner_close:]
            replacements.append((s_abs, e_abs, new_ns_block))
            dbg(f"  queued namespace replacement [{s_abs},{e_abs}] -> new_len={len(new_ns_block)}")
            if defs_for_ns:
                wrapped = f"namespace {ns_name} {{\n\n" + "\n\n".join(defs_for_ns) + "\n\n}\n"
                dbg(f"  Wrapping {len(defs_for_ns)} defs into namespace '{ns_name}' (len={len(wrapped)})")
                collected_out_of_line_defs.append(wrapped)

        # mask replaced spans so top-level class detection won't re-capture inner classes
        masked = list(text)
        for s,e,_ in replacements:
            for i in range(s, e+1):
                masked[i] = " "
        masked_text = "".join(masked)
        class_spans_abs = extract_blocks_positions(masked_text, CLASS_RE)
        dbg(f"Top-level class spans (absolute): {class_spans_abs}")
        for (s_abs, e_abs) in class_spans_abs:
            class_text = text[s_abs:e_abs+1]
            dbg(f"Processing top-level class at [{s_abs},{e_abs}] len={len(class_text)}")
            new_class_text, defs = process_class_block_for_out_of_line(class_text, namespace_prefix="")
            replacements.append((s_abs, e_abs, new_class_text))
            dbg(f"  queued class replacement [{s_abs},{e_abs}] -> new_len={len(new_class_text)} defs={len(defs)}")
            if defs:
                collected_out_of_line_defs.extend(defs)

        dbg(f"Total replacements queued: {len(replacements)}")
        new_text_with_repl = apply_replacements_to_text(text, replacements)

        # find free top-level functions (excluding occupied spans)
        occupied = [(s,e) for (s,e,_) in replacements]
        occupied_sorted = sorted(occupied)
        def is_occupied(pos):
            for s,e in occupied_sorted:
                if s <= pos <= e:
                    return True
            return False

        free_func_spans = []
        i = 0
        L = len(new_text_with_repl)
        while i < L:
            if is_occupied(i):
                for s,e in occupied_sorted:
                    if s <= i <= e:
                        i = e + 1
                        break
                continue
            paren = new_text_with_repl.find("(", i)
            if paren == -1:
                break
            line_start = new_text_with_repl.rfind("\n", 0, paren) + 1
            brace = new_text_with_repl.find("{", paren)
            if brace == -1:
                i = paren + 1
                continue
            close = find_matching_brace(new_text_with_repl, brace)
            if close == -1:
                i = paren + 1
                continue
            if is_occupied(brace):
                i = close + 1
                continue
            free_func_spans.append((line_start, close))
            dbg(f"Found free-function candidate span [{line_start},{close}] len={close-line_start+1}")
            i = close + 1

        dbg(f"Total free-function candidates found (top-level): {len(free_func_spans)}")
        free_repls_for_header = []
        source_defs = []
        for (fs, fe) in free_func_spans:
            func_text = new_text_with_repl[fs:fe+1]
            brace_pos = func_text.find("{")
            if brace_pos == -1:
                continue
            signature = func_text[:brace_pos].strip()
            if "::" in signature.split("(")[0]:
                dbg(f"Candidate appears qualified ('::') -> keep full def in source: '{signature[:120]}'")
                source_defs.append(func_text)
                free_repls_for_header.append((fs, fe, ""))  # remove from header
            else:
                decl = signature.rstrip() + ";"
                dbg(f"Converting free function signature to declaration for header: '{decl[:120]}'")
                free_repls_for_header.append((fs, fe, decl))
                source_defs.append(func_text)

        header_text = apply_replacements_to_text(new_text_with_repl, free_repls_for_header)

        header_final = "#pragma once\n\n" + (header_includes_text + "\n" if header_includes_text else "") + header_text.strip() + "\n"
        src_parts = [f'#include "{basename}.hpp"\n\n']
        if collected_out_of_line_defs:
            dbg(f"Appending {len(collected_out_of_line_defs)} collected out-of-line defs to source")
            src_parts.append("\n\n".join(collected_out_of_line_defs))
            src_parts.append("\n\n")
        if source_defs:
            src_parts.append("\n\n".join(source_defs))
        source_final = "".join(src_parts).strip() + "\n"

        header_final = re.sub(r'\n{3,}', '\n\n', header_final)
        header_final = re.sub(r';\s*;', ';', header_final)
        source_final = re.sub(r'\n{3,}', '\n\n', source_final)
        source_final = re.sub(r';\s*;', ';', source_final)

        return header_final, source_final

    except Exception as ex:
        warn("splitter failed: " + str(ex))
        warn(traceback.format_exc())
        fallback_header = "#pragma once\n\n" + processed_text
        fallback_source = f'#include "{basename}.hpp"\n\n// Fallback: could not split reliably\n'
        return fallback_header, fallback_source

# ---------------------------
# Supporting helpers (used earlier)
# ---------------------------
def extract_top_includes(text: str) -> Tuple[List[Tuple[int,int,str]], str]:
    lines = text.splitlines(keepends=True)
    includes = []
    i = 0
    for idx, ln in enumerate(lines):
        if re.match(r'^\s*#include\s+["<].+[">]', ln):
            start = sum(len(l) for l in lines[:idx])
            end = start + len(ln) - 1
            includes.append((start, end, ln))
            i = idx + 1
        elif ln.strip() == "":
            i = idx + 1
            continue
        else:
            break
    remaining = "".join(lines[i:])
    return includes, remaining

# ---------------------------
# File discovery / main pipeline
# ---------------------------
def gather_source_files(target_dir: str, ext: str, recursive: bool=True) -> List[str]:
    out = []
    if recursive:
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                if f.endswith("." + ext):
                    out.append(os.path.join(root, f))
    else:
        for f in os.listdir(target_dir):
            if f.endswith("." + ext) and os.path.isfile(os.path.join(target_dir, f)):
                out.append(os.path.join(target_dir, f))
    out.sort()
    return out

def ensure_dirs():
    os.makedirs(BUILD_HEADER_DIR, exist_ok=True)
    os.makedirs(BUILD_SRC_DIR, exist_ok=True)

def process_all():
    info(f"Starting preprocessing (ext=.{FILE_EXT}, target_dir='{TARGET_DIR}', attrs='{ATTR_DIR}', recursive={RECURSIVE})")
    funcs = load_functions_from_dir(ATTR_DIR)
    info(f"Loaded attribute functions: {list(funcs.keys())}")
    meta = load_edit_meta(META_FILE)
    files = gather_source_files(TARGET_DIR, FILE_EXT, recursive=RECURSIVE)
    info(f"Discovered {len(files)} files with extension .{FILE_EXT}")
    toproc = []
    for fp in files:
        rel = os.path.relpath(fp, TARGET_DIR)
        try:
            mtime = os.path.getmtime(fp)
        except Exception:
            mtime = time.time()
        recorded = meta.get(rel)
        if recorded is None or mtime > recorded:
            toproc.append((fp, rel, mtime))
    info(f"{len(toproc)} files need processing")
    if not toproc:
        info("Nothing to do.")
        return
    ensure_dirs()

    for fp, rel, mtime in toproc:
        info(f"--- Processing file: {fp}")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                src = f.read()
        except Exception as e:
            err(f"Failed to read '{fp}': {e}")
            continue

        try:
            processed = apply_attributes_to_text(src, funcs)
            info(f"Applied attributes (result len={len(processed)})")
        except KeyError as ke:
            err(f"Missing attribute function: {ke}")
            continue
        except Exception as e:
            err(f"Error during attribute application: {e}")
            err(traceback.format_exc())
            continue

        processed = rewrite_includes(processed, FILE_EXT)

        base = os.path.splitext(os.path.basename(fp))[0]
        try:
            header_text, source_text = split_header_and_source_verbose(processed, base)
        except Exception as e:
            err(f"Split failed for '{fp}': {e}")
            err(traceback.format_exc())
            header_text = "#pragma once\n\n" + processed
            source_text = f'#include "{base}.hpp"\n\n// split failed\n'

        if not header_text.strip():
            warn("Header empty after split -> fallback")
            header_text = "#pragma once\n\n" + processed
        if not source_text.strip():
            warn("Source empty after split -> using header include stub")
            source_text = f'#include "{base}.hpp"\n\n// no definitions extracted\n'

                # --- write outputs preserving folder structure relative to TARGET_DIR ---
        rel_dir = os.path.dirname(rel)  # e.g. "proj" or "" for root
        out_header_dir = os.path.join(BUILD_HEADER_DIR, rel_dir)
        out_src_dir = os.path.join(BUILD_SRC_DIR, rel_dir)
        os.makedirs(out_header_dir, exist_ok=True)
        os.makedirs(out_src_dir, exist_ok=True)

        header_fp = os.path.join(out_header_dir, f"{base}.hpp")
        src_fp = os.path.join(out_src_dir, f"{base}.cpp")

        try:
            with open(header_fp, "w", encoding="utf-8") as hf:
                hf.write(header_text)
            with open(src_fp, "w", encoding="utf-8") as sf:
                sf.write(source_text)
            info(f"Wrote: {header_fp} ({len(header_text)} bytes), {src_fp} ({len(source_text)} bytes)")
            meta[rel] = mtime
        except Exception as e:
            err(f"Failed to write outputs for '{fp}': {e}")
            err(traceback.format_exc())
            continue


    write_edit_meta(META_FILE, meta)
    info(f"Updated edit.meta at '{META_FILE}'")

if __name__ == "__main__":
    process_all()
