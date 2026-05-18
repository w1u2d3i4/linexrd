"""Reorganize main.tex section structure to a clean canonical layout.

Strategy: parse all (sub)section blocks, keep their bodies, then re-emit in
the target hierarchy. Merges and drops are explicit below.
"""
from __future__ import annotations
import re
import pathlib

P = pathlib.Path('/opt/data/private/linexrd/paper/main.tex')
src = P.read_text()

# -- isolate preamble (before \section{Introduction}) and back-matter --------
m_intro = re.search(r'\n\\section\{Introduction\}', src)
assert m_intro, 'no Introduction found'
preamble = src[:m_intro.start() + 1]   # keep leading \n
after_intro = src[m_intro.start():]

# real order in this file: ... §Conclusion ... \begin{thebibliography}...\end{thebibliography}
# \appendix ... \end{document}
m_bib = re.search(r'\n\\begin\{thebibliography\}', after_intro)
assert m_bib, 'no thebibliography'
body_text = after_intro[:m_bib.start()]
after_bib = after_intro[m_bib.start():]

m_end_bib = re.search(r'\\end\{thebibliography\}', after_bib)
assert m_end_bib, 'no \\end{thebibliography}'

m_appendix = re.search(r'\n\\appendix\b', after_bib)
assert m_appendix, 'no \\appendix'
bib_block = after_bib[:m_appendix.start()]   # includes thebibliography
appendix_and_end = after_bib[m_appendix.start():]

m_end_doc = re.search(r'\\end\{document\}', appendix_and_end)
appendix_only = appendix_and_end[:m_end_doc.start()]
end_doc = appendix_and_end[m_end_doc.start():]

# We will keep "bib_block" + "appendix" + "\\end{document}", but place
# the appendix BEFORE the bibliography per IJWMIP convention.
bib_and_after = bib_block + '\n' + end_doc


def split_blocks(text: str, level: str) -> list[tuple[str, str]]:
    """Return list of (title, body) for top-level \\<level>{...} chunks."""
    pat = re.compile(rf'\\{level}\*?\{{([^}}]+)\}}')
    matches = list(pat.finditer(text))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group(1).strip()
        # body starts right after the closing brace of the \section{...}
        body = text[m.end():end].lstrip('\n')
        out.append((title, body))
    return out


# ---------- harvest all subsections by their parent section ----------------
sections = split_blocks(body_text, 'section')


def harvest_subsections(sec_body: str) -> tuple[str, dict[str, str]]:
    """Return (intro_text_before_first_subsection, {title: body})."""
    subpat = re.compile(r'\\subsection\*?\{([^}]+)\}')
    matches = list(subpat.finditer(sec_body))
    if not matches:
        return sec_body, {}
    intro = sec_body[:matches[0].start()].strip()
    out = {}
    for i, mm in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(sec_body)
        ttl = mm.group(1).strip()
        body = sec_body[mm.end():end].strip()
        out[ttl] = body
    return intro, out


bank: dict[str, tuple[str, dict[str, str]]] = {}   # section title -> (intro, subs)
for sec_title, sec_body in sections:
    intro, subs = harvest_subsections(sec_body)
    bank[sec_title] = (intro, subs)

# ---------- harvest appendix sections --------------------------------------
appendix_sections = split_blocks(appendix_only, 'section')


def get(sec: str, sub: str) -> str:
    return bank[sec][1].get(sub, '')


# ---------- compose the new body ------------------------------------------
def emit_section(title: str, blocks: list[tuple[str, str]]) -> str:
    out = [f'\n\\section{{{title}}}\n\\label{{sec:{re.sub(r"[^a-zA-Z0-9]", "_", title.lower())}}}']
    for sub_title, sub_body in blocks:
        if sub_title:
            out.append(f'\n\\subsection{{{sub_title}}}')
        if sub_body.strip():
            out.append(sub_body.strip())
    return '\n\n'.join(out) + '\n'


new_body = []

# 1. Introduction (unchanged inner order, drop "The leaderboard question")
intro_section = bank['Introduction']
intro_blocks = [
    ('', intro_section[0]),
    ('The signal: 1-D, long, sparse, multi-resolution',
     get('Introduction', 'The signal: 1-D, long, sparse, multi-resolution')),
    ('Two physical constraints on PXRD representation learning',
     get('Introduction', 'Two physical constraints on PXRD representation learning')),
    ('Linear-time sequence models meet multi-resolution PXRD',
     get('Introduction', 'Linear-time sequence models meet multi-resolution PXRD')),
    ('Contributions', get('Introduction', 'Contributions')),
    ('Paper organisation', get('Introduction', 'Paper organisation')),
]
new_body.append(emit_section('Introduction', intro_blocks))

# 2. Related Work
rw_section = bank['Related Work']
rw_blocks = [
    ('', rw_section[0]),
    ('Multi-resolution signal analysis and wavelet stems',
     get('Related Work', 'Multi-resolution signal analysis and 1-D scientific signals')),
    ('Deep learning for PXRD space-group classification',
     get('Related Work', 'Deep learning for PXRD space-group classification')),
    ('Linear-time sequence models',
     get('Related Work', 'Linear-time sequence models') + '\n\n' +
     get('Related Work', 'Theoretical analysis of linear-time sequence models')),
    ('Hierarchical classification',
     get('Related Work', 'Hierarchical classification')),
]
new_body.append(emit_section('Related Work', rw_blocks))

# 3. Preliminaries  --  inject empirical spectrum here (was misplaced in §6)
empirical_spectrum = get('Experiments', 'Empirical spectrum of the PXRD signal')
prelim_section = bank['Preliminaries']
prelim_blocks = [
    ('', prelim_section[0]),
    ('PXRD physics in one paragraph',
     get('Preliminaries', 'PXRD physics in one paragraph')),
    ('Multi-resolution structure of the PXRD signal',
     get('Preliminaries', 'Multi-resolution structure of the PXRD signal') + '\n\n' +
     '\\paragraph{Empirical power spectrum.}\n' + empirical_spectrum),
    ('Space-group hierarchy', get('Preliminaries', 'Space-group hierarchy')),
    ('Linear-time encoders in formal one-liner each',
     get('Preliminaries', 'Linear-time encoders in formal one-liner each')),
]
new_body.append(emit_section('Preliminaries', prelim_blocks))

# 4. Data (unchanged)
data_section = bank['Data']
data_blocks = [('', data_section[0])] + [(t, b) for t, b in data_section[1].items()]
new_body.append(emit_section('Data', data_blocks))

# 5. Method  --  consolidate to 6 subsections in proper order
method = bank['Method'][1]
method_blocks = [
    ('', bank['Method'][0]),
    ('Multi-resolution convolutional stem',
     method.get('Multi-resolution convolutional stem', '') + '\n\n' +
     method.get('Multi-resolution analysis of the stem', '')),
    ('Encoder variants', method.get('Encoder variants', '')),
    ('Classification head and loss',
     method.get('Classification head', '') + '\n\n' +
     method.get('Loss formulation', '')),
    ('Training protocol',
     method.get('Training', '') + '\n\n' +
     method.get('Algorithmic pseudocode', '')),
    ('Evaluation metrics and cost--accuracy frontier',
     method.get('Metrics', '') + '\n\n' +
     method.get('Cost--accuracy frontier', '')),
    ('Computational complexity',
     method.get('Computational complexity in big-O', '') + '\n\n' +
     method.get('Why standard attention is not even on the table', '')),
]
new_body.append(emit_section('Method', method_blocks))

# 6. Experiments  --  drop misplaced spectrum & frontier-figure stub,
#                    inject learned-filter spectral analysis (was in §7)
exp = bank['Experiments'][1]
disc = bank['Discussion'][1]
spectral_filters = disc.get('Spectral analysis of learned filters', '')
exp_blocks = [
    ('', bank['Experiments'][0]),
    ('Main result: four-backbone frontier on CCDC-SG',
     exp.get('Main result: four-backbone frontier on CCDC-SG', '') + '\n\n' +
     exp.get('Cost--accuracy frontier figure', '')),
    ('Ablation: input resolution',
     exp.get('Ablation 1: Multi-resolution input ablation\n(full-resolution vs.\\ down-sampled)', '') or
     exp.get('Ablation 1: Multi-resolution input ablation', '')),
    ('Ablation: hierarchical CS$\\to$SG head',
     exp.get('Ablation 2: Hierarchical CS$\\to$SG consistency head', '')),
    ('Ablation: cross-dataset transfer (INOR, CoREMOF19)',
     exp.get('Ablation 3: Cross-dataset transfer (INOR, CoREMOF19)', '')),
    ('Ablation: Mamba-2 depth/width sweep',
     exp.get('Ablation 4: Mamba-2 depth/width sweep', '')),
    ('Ablation: hyperparameter sensitivity',
     exp.get('Ablation 5: Hyperparameter sensitivity', '')),
    ('Spectral analysis of learned filters', spectral_filters),
    ('Failure-mode analysis', exp.get('Failure-mode analysis', '')),
]
new_body.append(emit_section('Experiments', exp_blocks))

# 7. Discussion  --  consolidate to 8 subsections
disc_blocks = [
    ('', bank['Discussion'][0]),
    ('Why does Mamba-2 win at full resolution?',
     disc.get('Why does Mamba-2 win at full resolution?', '')),
    ('Why is GLA an attractive compromise?',
     disc.get('Why is GLA an attractive compromise?', '')),
    ('Why is RetNet Pareto-dominated?',
     disc.get('Why is RetNet Pareto-dominated?', '')),
    ('Multi-resolution and compressed-sensing interpretation',
     disc.get('Multi-resolution interpretation', '') + '\n\n' +
     disc.get('Connection to compressed sensing', '')),
    ('Implications for materials informatics',
     disc.get('Implications for materials informatics', '') + '\n\n' +
     disc.get('Implications for autonomous materials laboratories', '')),
    ('Robustness analysis', disc.get('Robustness analysis', '')),
    ('Limitations and threats to validity',
     disc.get('Limitations', '') + '\n\n' +
     disc.get('Threats to validity', '')),
    ('Future work', disc.get('Future work', '')),
]
new_body.append(emit_section('Discussion', disc_blocks))

# 8. Conclusion (preserve as-is, single \section)
conc_intro = bank['Conclusion'][0]
new_body.append(f'\n\\section{{Conclusion}}\n\\label{{sec:conclusion}}\n\n{conc_intro}\n')

# Acknowledgements section (starred)
ack = bank.get('Acknowledgements', ('', {}))[0]
if ack:
    new_body.append(f'\n\\section*{{Acknowledgements}}\n\n{ack}\n')


# ---------- compose the trimmed appendix -----------------------------------
ax_dict = {t: b for t, b in appendix_sections}

new_appendix = ['\n\\appendix\n']
# A. Hyperparameters
new_appendix.append(f'\n\\section{{Hyperparameters in full}}\n\\label{{app:a}}\n\n{ax_dict.get("Hyperparameters in full","").strip()}\n')
# B. Notation glossary
new_appendix.append(f'\n\\section{{Notation glossary}}\n\\label{{app:b}}\n\n{ax_dict.get("Notation glossary","").strip()}\n')
# C. Multi-resolution lemma proof
new_appendix.append(f'\n\\section{{Proof of the multi-resolution lemma}}\n\\label{{app:c}}\n\n{ax_dict.get("Proof of the multi-resolution lemma","").strip()}\n')
# D. Reproducibility checklist (merge with old "Computational reproducibility")
repro_body = ax_dict.get('Reproducibility checklist','').strip()
comp_repro_body = ax_dict.get('Computational reproducibility','').strip()
new_appendix.append(
    f'\n\\section{{Reproducibility checklist}}\n\\label{{app:d}}\n\n{comp_repro_body}\n\n{repro_body}\n'
)
# E. Wavelet-stem comparison + learned filter spectra (full)
new_appendix.append(
    f'\n\\section{{Wavelet-stem comparison and learned filter spectra}}\n\\label{{app:e}}\n\n'
    f'{ax_dict.get("Wavelet-stem comparison and spectrum of learned filters","").strip()}\n'
)

# (drop the placeholder stubs:
#   "40-epoch results", "Sim-to-real on RRUFF",
#   "S4 / S5 preliminary results", "Smoke-run training curves",
#   "Per-class confusion-matrix slices", "Negative-result protocol")

# ---------- write ----------------------------------------------------------
new_text = preamble + '\n'.join(new_body) + '\n' + ''.join(new_appendix) + '\n' + bib_and_after
P.write_text(new_text)

# count
print(f'Wrote new main.tex: {len(new_text.split())} words')
print('Section count after reorg:',
      len(re.findall(r'\\section\*?\{', new_text)))
print('Subsection count after reorg:',
      len(re.findall(r'\\subsection\*?\{', new_text)))
