#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Projektstruktúra TXT -> nagy felbontású PNG, kizárólag matplotlib-pel.

Nem járja be a mappákat, nem generál TXT-t/SVG-t, az inputot nem írja át.
A korábbi einn_structure_report.py formátumát és elrendezését követi.

Használat a projekt gyökérkönyvtárából:
    python einn_structure_report_png_matplotlib.py \
        --input project_structure.txt --output architecture_report --scale 2
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


@dataclass
class Node:
    name: str
    children: list['Node'] = field(default_factory=list)
    directory: bool = False


# Tree connectors accepted as well as indentation-only lists.
BRANCH = re.compile(r'^(?P<prefix>[ \t│|]*)(?:├──|└──|\|--|`--|\+--)[ \t]*(?P<name>.+?)\s*$')
TREE_SUMMARY = re.compile(r'^\s*\d+\s+director(?:y|ies),\s*\d+\s+files?\s*$', re.I)


def extract_entries(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Return (displayed project name, depth/name pairs) from the edited TXT."""
    lines = text.lstrip('\ufeff').splitlines()
    usable: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        # Older script appended a separate class inventory; it's not a tree.
        if line.lstrip().startswith('#'):
            break
        if TREE_SUMMARY.fullmatch(line):
            continue
        usable.append(line)
    if not usable:
        raise ValueError('A TXT nem tartalmaz ábrázolható fájlstruktúrát.')

    # The earlier exported tree added two spaces to every branch line. Find the
    # shared left margin; it should not count as hierarchy depth.
    branch_prefixes = [len(m.group('prefix').expandtabs(4))
                       for s in usable if (m := BRANCH.match(s))]
    common_margin = min(branch_prefixes) if branch_prefixes else 0

    raw_entries: list[tuple[int, str, bool]] = []
    for s in usable:
        m = BRANCH.match(s)
        if m:
            prefix = m.group('prefix').expandtabs(4)
            depth = max(0, (len(prefix) - common_margin) // 4)
            raw_entries.append((depth, m.group('name').strip(), True))
        else:
            stripped = s.strip()
            if not stripped:
                continue
            # Indentation-only input: infer the indentation step below.
            spaces = len(s.expandtabs(4)) - len(s.expandtabs(4).lstrip(' '))
            raw_entries.append((spaces, stripped, False))

    # Detect a root label from `tree` ('.', 'project/', etc.) followed by
    # top-level branch entries. It is a label, not a drawn directory child.
    project_name = 'Projektstruktúra'
    if len(raw_entries) > 1:
        first_depth, first_name, first_is_branch = raw_entries[0]
        second_depth, _, second_is_branch = raw_entries[1]
        if (not first_is_branch and second_is_branch and second_depth == 0):
            project_name = first_name.rstrip('/') if first_name != '.' else 'Projekt'
            raw_entries.pop(0)
        elif first_name == '.' and not first_is_branch:
            project_name = 'Projekt'
            raw_entries.pop(0)

    indents = sorted({depth for depth, _, br in raw_entries if not br and depth > 0})
    unit = indents[0] if indents else 2
    result: list[tuple[int, str]] = []
    for depth, name, was_branch in raw_entries:
        # Indentation-only levels become 0,1,2,...; branch levels already are.
        if not was_branch:
            depth //= unit
        result.append((depth, name))

    # If the TXT is entirely an ordinary indented tree, the first line may be
    # the project root. Keep it as part of the tree unless there is a clear
    # `tree`-style root label.
    return project_name, result


def parse_tree(text: str) -> tuple[str, list[Node]]:
    project_name, entries = extract_entries(text)
    roots: list[Node] = []
    stack: list[Node] = []
    for depth, name in entries:
        if not name:
            continue
        # A child implies its parent is a directory, whether or not it has '/' .
        depth = min(depth, len(stack))  # tolerate accidental indentation jumps
        node = Node(name=name, directory=name.endswith('/'))
        while len(stack) > depth:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
            stack[-1].directory = True
        else:
            roots.append(node)
        stack.append(node)
    return project_name, roots


def render_children(nodes: list[Node], prefix: str = '') -> list[tuple[str, bool]]:
    lines: list[tuple[str, bool]] = []
    for idx, n in enumerate(nodes):
        last = idx == len(nodes) - 1
        connector = '└── ' if last else '├── '
        lines.append((prefix + connector + n.name, n.directory))
        lines.extend(render_children(n.children, prefix + ('    ' if last else '│   ')))
    return lines


def make_groups(roots: list[Node]) -> list[tuple[str, list[tuple[str, bool]]]]:
    groups: list[tuple[str, list[tuple[str, bool]]]] = []
    root_files: list[Node] = []
    for node in roots:
        if node.directory:
            groups.append((node.name, render_children(node.children)))
        else:
            root_files.append(node)
    if root_files:
        pass #groups.append(('Gyökérfájlok / entry pointok', render_children(root_files)))
    return groups or [('(üres struktúra)', [])]




def write_png(project_name: str, roots: list[Node], target: Path,
              title: str, columns: int, scale: float) -> None:
    """Ugyanazok a csoportok/kártyák, mint a korábbi SVG-ben; PNG közvetlenül."""
    groups = make_groups(roots)
    slots: list[list[tuple[str, list[tuple[str, bool]]]]] = [[] for _ in range(columns)]
    heights = [0] * columns
    for heading, lines in groups:
        i = min(range(columns), key=lambda j: heights[j])
        slots[i].append((heading, lines))
        heights[i] += len(lines) + 3

    base_dpi = 100
    pad = 34
    line_height = 28
    title_height = 128
    widths = []
    for groups_in_col in slots:
        texts = [t for _, lines in groups_in_col for t, _ in lines]
        texts.extend(heading for heading, _ in groups_in_col)
        max_chars = max([len(t) for t in texts] + [28])
        widths.append(max(670, int(max_chars * 11 + 110)))

    width = sum(widths) + (columns + 1) * pad
    height = max(370, title_height + max(heights, default=0) * line_height + 80)

    # A koordinátarendszer pixelekben működik. scale csak a mentett
    # kép felbontását növeli, a feliratok/elrendezés arányát nem.
    fig = plt.figure(figsize=(width / base_dpi, height / base_dpi), dpi=base_dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")

    def txt(x: float, y: float, s: str, *, size_px: float,
            color: str, bold: bool = False, mono: bool = False):
        # 100 dpi alapméreten a px -> pt átváltás: * 72 / 100.
        ax.text(x, y, s, va="baseline", ha="left", color=color,
                fontfamily="DejaVu Sans Mono" if mono else "DejaVu Sans",
                fontsize=size_px * 72 / base_dpi,
                fontweight="bold" if bold else "normal",
                clip_on=False)

    txt(pad, 56, title, size_px=31, color="#18263b", bold=True)
    txt(pad, 88, f"{project_name} · az input TXT alapján",
        size_px=16, color="#58677a")

    x = pad
    for ci, col_groups in enumerate(slots):
        w = widths[ci]
        ax.add_patch(FancyBboxPatch((x, 107), w, height - 130,
                       boxstyle="round,pad=0,rounding_size=14",
                       facecolor="white", edgecolor="#d7e0ec", linewidth=1.5))
        y = 153
        for heading, lines in col_groups:
            ax.add_patch(FancyBboxPatch((x + 16, y - 29), w - 32, 40,
                           boxstyle="round,pad=0,rounding_size=8",
                           facecolor="#e9f1ff", edgecolor="none"))
            txt(x + 28, y - 1, heading, size_px=21, color="#224c9c", bold=True)
            y += 38
            for name, is_dir in lines:
                txt(x + 24, y, name, size_px=17,
                    color="#1d559b" if is_dir else "#283649",
                    bold=is_dir, mono=True)
                y += line_height
            y += 22
        x += w + pad

    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format="png", dpi=base_dpi * scale,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meglévő TXT projektfa -> PNG, matplotlib; semmi SVG / új TXT.")
    parser.add_argument("--input", default="project_structure.txt",
                        help="A kézzel szerkesztett TXT fájl útvonala")
    parser.add_argument("--output", default="architecture_report",
                        help="Kimeneti könyvtár VAGY közvetlen .png fájlnév")
    parser.add_argument("--title", default="EINN – projektstruktúra")
    parser.add_argument("--columns", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--scale", type=float, default=2.0,
                        help="Képfelbontás szorzó (alapértelmezés 2)")
    args = parser.parse_args()
    if not (0.5 <= args.scale <= 5):
        parser.error("A --scale értéke 0.5 és 5 között legyen.")

    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Nem található a bemeneti TXT: {source}")
    dest = Path(args.output).expanduser().resolve()
    if dest.suffix.lower() != ".png":
        dest = dest / "project_structure.png"
    name, roots = parse_tree(source.read_text(encoding="utf-8-sig"))
    write_png(name, roots, dest, args.title, args.columns, args.scale)
    print(f"Elkészült: {dest}")
    print("A TXT változatlan, SVG nem keletkezett.")


if __name__ == "__main__":
    main()