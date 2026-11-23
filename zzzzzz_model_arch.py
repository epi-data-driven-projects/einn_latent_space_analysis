#!/usr/bin/env python3
"""EINN project tree visualizer: reads an existing TXT, produces only an SVG.

The TXT is the single source of truth: no repository scanning, no TXT output,
no AST parsing, and no inferred files/classes.

Example (from the project directory):
    python3 einn_structure_report.py --input project_structure.txt --output architecture_report

The input may come from `tree`, the earlier report script, or a manually edited
file, with Unicode tree branches (├──, └──), ASCII branches (|--, `--), or
ordinary indentation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import html
from pathlib import Path
import re


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
    project_name = 'Struktúra'
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
        groups.append(('Gyökérfájlok', render_children(root_files)))
    return groups or [('(üres struktúra)', [])]


def write_svg(project_name: str, roots: list[Node], target: Path,
              title: str, columns: int) -> None:
    groups = make_groups(roots)
    columns = max(1, columns)
    slots: list[list[tuple[str, list[tuple[str, bool]]]]] = [[] for _ in range(columns)]
    heights = [0] * columns
    # Maintain the TXT's top-level order within each column.
    for name, lines in groups:
        i = min(range(columns), key=lambda j: heights[j])
        slots[i].append((name, lines))
        heights[i] += len(lines) + 3

    font_size = 17
    line_h = 28
    title_h = 128
    pad = 34
    # Width grows with the longest tree line, so even a deep tree is visible.
    widths: list[int] = []
    for group_list in slots:
        texts = [line for _, lines in group_list for line, _ in lines]
        texts += [name for name, _ in group_list]
        max_chars = max([len(t) for t in texts] + [28])
        widths.append(max(670, int(max_chars * 11 + 110)))
    total_width = sum(widths) + (columns + 1) * pad
    max_height = max(heights, default=0)
    total_height = max(370, title_h + max_height * line_h + 80)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        f'<text x="{pad}" y="56" style="font:700 31px sans-serif;fill:#18263b">{html.escape(title)}</text>',
    ]

    x = pad
    for ci, groups_in_col in enumerate(slots):
        w = widths[ci]
        parts.append(f'<rect x="{x}" y="107" width="{w}" height="{total_height - 130}" rx="14" fill="white" stroke="#d7e0ec" stroke-width="1.5"/>')
        y = 153
        for heading, lines in groups_in_col:
            parts.append(f'<rect x="{x+16}" y="{y-29}" width="{w-32}" height="40" rx="8" fill="#e9f1ff"/>')
            parts.append(f'<text x="{x+28}" y="{y-1}" style="font:700 21px sans-serif;fill:#224c9c">{html.escape(heading)}</text>')
            y += 38
            for name, directory in lines:
                color = '#1d559b' if directory else '#283649'
                weight = 600 if directory else 400
                parts.append(f'<text x="{x+24}" y="{y}" xml:space="preserve" style="font:{weight} {font_size}px monospace;fill:{color}">{html.escape(name)}</text>')
                y += line_h
            y += 22
        x += w + pad

    parts.append('</svg>')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(parts) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Meglévő TXT faábra -> SVG (csak olvasás a TXT-ből).')
    parser.add_argument('--input', default='project_structure.txt',
                        help='A már megszerkesztett, létező TXT elérési útja')
    parser.add_argument('--output', default='architecture_report',
                        help='Kimeneti mappa, vagy .svg végű kimeneti fájl')
    parser.add_argument('--title', default='EINN – Struktúra')
    parser.add_argument('--columns', type=int, choices=[1, 2, 3], default=2)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        parser.error(f'Nem található a bemeneti TXT: {source}')
    destination = Path(args.output).expanduser().resolve()
    if destination.suffix.lower() != '.svg':
        destination = destination / 'project_structure.svg'

    project_name, nodes = parse_tree(source.read_text(encoding='utf-8-sig'))
    write_svg(project_name, nodes, destination, args.title, args.columns)
    print(f'Az ábra elkészült: {destination}')
    print('A bemeneti TXT változatlan maradt; nem készült új TXT vagy más jelentés.')


if __name__ == '__main__':
    main()