"""Render editable TikZ diagrams without changing model data or numerical code.

Example (the project's existing portable environment):
    python scripts/render_paper_diagrams.py --export-relation
The main paper inputs the TikZ sources directly and does not need this helper.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HEIGHTS = {'relation': 74, 'geometry': 62, 'fvm': 77, 'solver': 128}
PREAMBLE = r'''\documentclass[12pt]{ctexart}
\usepackage[paperwidth=164mm,paperheight=HEIGHTmm,margin=2mm]{geometry}
\usepackage{amsmath,amssymb,xcolor,tikz}
\usetikzlibrary{arrows.meta,positioning,calc,shapes.geometric,fit}
\input{SOURCES/style}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\begin{document}
\noindent\input{SOURCES/NAME}
\end{document}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, help='Path to XeLaTeX or Tectonic')
    parser.add_argument('--outdir', type=Path, default=ROOT / '_tmp/visual_upgrade/diagram_previews')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '_tmp/visual_upgrade/tectonic-cache')
    parser.add_argument('--bundle', default='http://127.0.0.1:2481/bundle.tar')
    parser.add_argument('--export-relation', action='store_true', help='Update the standalone relation PDF/PNG assets')
    args = parser.parse_args()
    portable = ROOT / '_tmp/post_contest/tectonic-0.17.0/tectonic.exe'
    engine = args.engine or (Path(shutil.which('xelatex')) if shutil.which('xelatex') else portable)
    if not engine.is_file():
        parser.error('Provide --engine with an available XeLaTeX or Tectonic executable.')
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = Path(os.path.relpath(ROOT / 'paper/diagrams', out)).as_posix()
    env = os.environ.copy()
    env['TECTONIC_CACHE_DIR'] = str(args.cache_dir.resolve())
    for name, height in HEIGHTS.items():
        tex = PREAMBLE.replace('HEIGHT', str(height)).replace('SOURCES', sources).replace('NAME', name)
        source = out / f'{name}.tex'
        source.write_text(tex, encoding='utf-8')
        if 'tectonic' in engine.name.lower():
            cmd = [str(engine), '-X', 'compile', str(source), '--bundle', args.bundle,
                   '--outdir', str(out), '--keep-logs', '--only-cached']
        else:
            cmd = [str(engine), '-interaction=nonstopmode', '-halt-on-error',
                   '-output-directory', str(out), str(source)]
        run = subprocess.run(cmd, cwd=out, env=env, capture_output=True)
        (out / f'{name}-console.log').write_bytes(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f'{name}: compilation failed; inspect {name}-console.log')
        raster = subprocess.run(['pdftoppm', '-png', '-singlefile', '-r', '160',
                                 str(out / f'{name}.pdf'), str(out / name)], capture_output=True)
        (out / f'{name}-raster.log').write_bytes(raster.stdout + raster.stderr)
        raster.check_returncode()
        print(f'Rendered {name}: {out / (name + ".pdf")}')
    if args.export_relation:
        for suffix in ['pdf', 'png']:
            shutil.copyfile(out / f'relation.{suffix}', ROOT / f'paper/figures/fig_relation.{suffix}')
        print('Updated paper/figures/fig_relation.pdf and .png from the current TikZ source.')


if __name__ == '__main__':
    main()
