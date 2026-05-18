#!/usr/bin/env python3
"""
ppt_replica — convert a single PNG/PDF page (an infographic, paper figure,
slide, or schematic) into an editable .pptx whose text, panels, lines and
arrows are native PowerPoint objects, while complex chart/icon regions
are embedded as cropped PNG assets.

Inspired by ZhiweiWei-NAMI/PPT-Visual-Replica (Codex skill). This is the
Claude Code adaptation: same design philosophy, but Claude (not Codex)
authors a JSON spec by reading the source image, then this tool renders
the .pptx — no image-generation API call, no proprietary skill format.

Subcommands:
  extract    rasterize a PDF or copy/resize a PNG into a working dir
  crop       extract a sub-region of the source image as a PNG asset
  build      render a JSON spec into out.pptx
  measure    print pixel→inch conversion for a given canvas size
  pixelpick  print colors at requested pixel coords (lookup helper)

JSON spec format (one slide):

  {
    "slide": {"width_in": 13.333, "height_in": 7.5},
    "background": {
      "fill": "#FFFFFF",
      "image": "source.png",          # optional, drawn at full canvas
      "image_opacity": 0.0             # optional 0..1, ignored if no image
    },
    "elements": [
      { "kind": "panel",
        "x": 0.30, "y": 0.55, "w": 2.30, "h": 5.40,
        "fill": "#FFF7ED", "stroke": "#FB923C", "stroke_width": 2,
        "rounded": true, "corner_radius": 0.18,
        "title": {"text": "(1) Problem Structure",
                   "color": "#FB923C", "size": 18, "bold": true,
                   "align": "left", "pad_x": 0.18, "pad_y": 0.10}
      },
      { "kind": "text",
        "x": 0.4, "y": 1.2, "w": 2.0, "h": 0.4,
        "text": "Long-tail over space groups",
        "font": "Arial", "size": 12, "bold": true,
        "color": "#000000", "align": "center", "valign": "middle"
      },
      { "kind": "image",
        "x": 0.4, "y": 1.6, "w": 2.0, "h": 1.5,
        "src": "assets/longtail_chart.png"
      },
      { "kind": "arrow",
        "x1": 2.5, "y1": 3.5, "x2": 2.9, "y2": 3.5,
        "stroke": "#000000", "stroke_width": 2.0,
        "head_end": "triangle", "head_start": "none"
      },
      { "kind": "shape",
        "shape": "rounded_rect",      # rect | rounded_rect | oval | callout
        "x": 5.0, "y": 2.0, "w": 1.5, "h": 1.0,
        "fill": "#E0F2FE", "stroke": "#0284C7", "stroke_width": 1.5,
        "corner_radius": 0.10,
        "text": "label", "font_size": 11, "font_color": "#000",
        "align": "center", "valign": "middle"
      },
      { "kind": "line",
        "x1": 5.0, "y1": 6.0, "x2": 8.0, "y2": 6.0,
        "stroke": "#888", "stroke_width": 1.0, "dash": "solid"
      }
    ]
  }

Coords are in **inches**. Colors in `#RRGGBB`. The default canvas is
13.333" x 7.5" (16:9 widescreen).

Pure stdlib + python-pptx + Pillow. No network.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _require_pptx():
    try:
        import pptx  # noqa: F401
        return pptx
    except ImportError:
        print("[ppt_replica] python-pptx not installed; "
              "`pip install python-pptx Pillow`", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"bad hex color: {s}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _color(s: str | None):
    from pptx.dml.color import RGBColor
    if not s:
        return None
    return RGBColor(*_hex_to_rgb(s))


def _emu(inches: float):
    from pptx.util import Inches
    return Inches(inches)


# ---------------------------------------------------------------------------
# Element renderers
# ---------------------------------------------------------------------------

def _set_fill(shape, hexcolor: str | None) -> None:
    if hexcolor is None:
        shape.fill.background()
        return
    shape.fill.solid()
    shape.fill.fore_color.rgb = _color(hexcolor)


def _set_line(shape, hexcolor: str | None, width_pt: float | None,
               dash: str | None = None) -> None:
    from pptx.util import Pt
    line = shape.line
    if hexcolor is None:
        line.fill.background()
        return
    line.color.rgb = _color(hexcolor)
    if width_pt is not None:
        line.width = Pt(width_pt)
    # Dash
    if dash and dash != "solid":
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        m = {"dash": MSO_LINE_DASH_STYLE.DASH,
             "dot":  MSO_LINE_DASH_STYLE.ROUND_DOT,
             "dashdot": MSO_LINE_DASH_STYLE.DASH_DOT}
        if dash in m:
            line.dash_style = m[dash]


def _add_text_runs(text_frame, text: str, *, font: str | None,
                    size: float, bold: bool, italic: bool,
                    color: str | None, align: str | None,
                    valign: str | None) -> None:
    from pptx.util import Pt
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    text_frame.word_wrap = True
    text_frame.margin_left = _emu(0.04)
    text_frame.margin_right = _emu(0.04)
    text_frame.margin_top = _emu(0.02)
    text_frame.margin_bottom = _emu(0.02)
    if valign:
        m = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE,
             "bottom": MSO_ANCHOR.BOTTOM}
        if valign in m:
            text_frame.vertical_anchor = m[valign]
    # Allow \n line breaks
    paragraphs = text.split("\n")
    for i, para in enumerate(paragraphs):
        p = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
        if align:
            am = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER,
                   "right": PP_ALIGN.RIGHT, "justify": PP_ALIGN.JUSTIFY}
            if align in am:
                p.alignment = am[align]
        run = p.add_run()
        run.text = para
        run.font.size = Pt(size)
        if font:
            run.font.name = font
        run.font.bold = bool(bold)
        run.font.italic = bool(italic)
        if color:
            run.font.color.rgb = _color(color)


def _add_panel(slide, e: dict) -> None:
    from pptx.enum.shapes import MSO_SHAPE
    rounded = e.get("rounded", True)
    shape_kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(
        shape_kind, _emu(e["x"]), _emu(e["y"]), _emu(e["w"]), _emu(e["h"]),
    )
    if rounded and "corner_radius" in e:
        # Adjust the rounded-corner adjustment value (0..0.5).
        # python-pptx exposes adjustments[0] as relative to half min(w,h).
        try:
            min_in = min(e["w"], e["h"])
            adj = max(0.0, min(0.5, float(e["corner_radius"]) / min_in))
            s.adjustments[0] = adj
        except Exception:
            pass
    _set_fill(s, e.get("fill"))
    _set_line(s, e.get("stroke"), e.get("stroke_width", 1.5))
    # Title text
    title = e.get("title")
    if title:
        from pptx.util import Pt
        from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
        pad_x = title.get("pad_x", 0.18)
        pad_y = title.get("pad_y", 0.08)
        tw = max(0.6, e["w"] - 2 * pad_x)
        th = max(0.3, title.get("h", 0.42))
        tb = slide.shapes.add_textbox(
            _emu(e["x"] + pad_x), _emu(e["y"] + pad_y),
            _emu(tw), _emu(th),
        )
        _add_text_runs(
            tb.text_frame, title["text"],
            font=title.get("font", "Arial"),
            size=title.get("size", 16),
            bold=title.get("bold", True),
            italic=title.get("italic", False),
            color=title.get("color", "#000000"),
            align=title.get("align", "left"),
            valign=title.get("valign", "top"),
        )


def _add_text(slide, e: dict) -> None:
    tb = slide.shapes.add_textbox(
        _emu(e["x"]), _emu(e["y"]), _emu(e["w"]), _emu(e["h"]),
    )
    _add_text_runs(
        tb.text_frame, e["text"],
        font=e.get("font", "Arial"),
        size=e.get("size", 12),
        bold=e.get("bold", False),
        italic=e.get("italic", False),
        color=e.get("color", "#000000"),
        align=e.get("align", "left"),
        valign=e.get("valign", "top"),
    )


def _add_image(slide, e: dict, base_dir: Path) -> None:
    src = (base_dir / e["src"]).resolve()
    if not src.exists():
        print(f"[ppt_replica] WARN: image missing: {src}", file=sys.stderr)
        return
    slide.shapes.add_picture(
        str(src), _emu(e["x"]), _emu(e["y"]),
        _emu(e["w"]), _emu(e["h"]),
    )


def _add_shape(slide, e: dict) -> None:
    from pptx.enum.shapes import MSO_SHAPE
    name = e.get("shape", "rect")
    m = {
        "rect": MSO_SHAPE.RECTANGLE,
        "rounded_rect": MSO_SHAPE.ROUNDED_RECTANGLE,
        "oval": MSO_SHAPE.OVAL,
        "diamond": MSO_SHAPE.DIAMOND,
        "callout": MSO_SHAPE.RECTANGULAR_CALLOUT,
        "chevron": MSO_SHAPE.CHEVRON,
        "right_arrow": MSO_SHAPE.RIGHT_ARROW,
        "down_arrow": MSO_SHAPE.DOWN_ARROW,
        "left_arrow": MSO_SHAPE.LEFT_ARROW,
        "up_arrow": MSO_SHAPE.UP_ARROW,
        "pentagon": MSO_SHAPE.PENTAGON,
        "hexagon": MSO_SHAPE.HEXAGON,
        "parallelogram": MSO_SHAPE.PARALLELOGRAM,
        "trapezoid": MSO_SHAPE.TRAPEZOID,
        "circle": MSO_SHAPE.OVAL,
    }
    shape_kind = m.get(name, MSO_SHAPE.RECTANGLE)
    s = slide.shapes.add_shape(
        shape_kind, _emu(e["x"]), _emu(e["y"]), _emu(e["w"]), _emu(e["h"]),
    )
    if name == "rounded_rect" and "corner_radius" in e:
        try:
            min_in = min(e["w"], e["h"])
            adj = max(0.0, min(0.5, float(e["corner_radius"]) / min_in))
            s.adjustments[0] = adj
        except Exception:
            pass
    _set_fill(s, e.get("fill"))
    _set_line(s, e.get("stroke"), e.get("stroke_width", 1.0),
                e.get("dash", "solid"))
    if e.get("text"):
        _add_text_runs(
            s.text_frame, e["text"],
            font=e.get("font", "Arial"),
            size=e.get("font_size", e.get("size", 11)),
            bold=e.get("bold", False),
            italic=e.get("italic", False),
            color=e.get("font_color", e.get("color", "#000000")),
            align=e.get("align", "center"),
            valign=e.get("valign", "middle"),
        )


def _add_line(slide, e: dict, with_arrow: bool = False) -> None:
    from pptx.enum.shapes import MSO_CONNECTOR
    from pptx.util import Pt
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        _emu(e["x1"]), _emu(e["y1"]),
        _emu(e["x2"]), _emu(e["y2"]),
    )
    line = conn.line
    line.color.rgb = _color(e.get("stroke", "#000000"))
    line.width = Pt(e.get("stroke_width", 1.0))
    if e.get("dash") and e["dash"] != "solid":
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        m = {"dash": MSO_LINE_DASH_STYLE.DASH,
             "dot": MSO_LINE_DASH_STYLE.ROUND_DOT,
             "dashdot": MSO_LINE_DASH_STYLE.DASH_DOT}
        if e["dash"] in m:
            line.dash_style = m[e["dash"]]
    if with_arrow:
        # python-pptx does not expose arrowhead enum cleanly — write XML.
        from pptx.oxml.ns import qn
        from lxml import etree
        spPr = conn.line._get_or_add_ln()  # type: ignore[attr-defined]
        head_end = e.get("head_end", "triangle")
        head_start = e.get("head_start", "none")
        # Remove any existing tailEnd / headEnd
        for tag in (qn("a:headEnd"), qn("a:tailEnd")):
            for old in spPr.findall(tag):
                spPr.remove(old)
        if head_start and head_start != "none":
            he = etree.SubElement(spPr, qn("a:headEnd"))
            he.set("type", head_start)
            he.set("w", "med"); he.set("len", "med")
        if head_end and head_end != "none":
            te = etree.SubElement(spPr, qn("a:tailEnd"))
            te.set("type", head_end)
            te.set("w", "med"); te.set("len", "med")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(spec_path: Path, out_path: Path) -> None:
    _require_pptx()
    from pptx import Presentation
    from pptx.util import Inches

    spec = json.loads(spec_path.read_text())
    base_dir = spec_path.parent

    prs = Presentation()
    sl = spec.get("slide", {})
    w = sl.get("width_in", 13.333)
    h = sl.get("height_in", 7.5)
    prs.slide_width = Inches(w)
    prs.slide_height = Inches(h)

    blank_layout = prs.slide_layouts[6]  # 6 = blank
    slides_data = spec.get("slides")
    if slides_data is None:
        slides_data = [{"background": spec.get("background", {}),
                         "elements": spec.get("elements", [])}]

    for sl_data in slides_data:
        slide = prs.slides.add_slide(blank_layout)
        bg = sl_data.get("background", {})
        # Fill background by adding a rect spanning the slide
        if bg.get("fill"):
            from pptx.enum.shapes import MSO_SHAPE
            r = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height
            )
            _set_fill(r, bg["fill"])
            _set_line(r, None, 0)
        if bg.get("image"):
            img = (base_dir / bg["image"]).resolve()
            if img.exists():
                slide.shapes.add_picture(str(img), 0, 0,
                                          prs.slide_width, prs.slide_height)
        # Elements in order (z-order = call order)
        for e in sl_data.get("elements", []):
            kind = e.get("kind")
            try:
                if kind == "panel":
                    _add_panel(slide, e)
                elif kind == "text":
                    _add_text(slide, e)
                elif kind == "image":
                    _add_image(slide, e, base_dir)
                elif kind == "shape":
                    _add_shape(slide, e)
                elif kind == "line":
                    _add_line(slide, e, with_arrow=False)
                elif kind == "arrow":
                    _add_line(slide, e, with_arrow=True)
                else:
                    print(f"[ppt_replica] WARN: unknown kind '{kind}'",
                          file=sys.stderr)
            except Exception as ex:
                print(f"[ppt_replica] failed element {kind}: {ex}",
                      file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    print(json.dumps({"out": str(out_path),
                       "slides": len(slides_data)}, indent=2))


# ---------------------------------------------------------------------------
# Extract — PDF → PNGs
# ---------------------------------------------------------------------------

def extract(src: Path, out_dir: Path, dpi: int = 200) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        target = out_dir / f"page_001{src.suffix.lower()}"
        shutil.copy2(src, target)
        print(json.dumps({"pages": [str(target)]}, indent=2))
        return
    if src.suffix.lower() != ".pdf":
        raise SystemExit(f"unsupported input: {src.suffix}")
    # Use pdftoppm to one PNG per page
    prefix = out_dir / "page"
    cmd = ["pdftoppm", "-png", "-r", str(dpi), str(src), str(prefix)]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit("pdftoppm failed")
    pages = sorted(out_dir.glob("page-*.png"))
    print(json.dumps({"pages": [str(p) for p in pages]}, indent=2))


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

def crop(src: Path, out: Path, box: tuple[int, int, int, int],
          *, transparent_white: bool = False,
          tolerance: int = 8) -> None:
    from PIL import Image
    im = Image.open(src).convert("RGBA")
    left, top, right, bottom = box
    sub = im.crop((left, top, right, bottom))
    if transparent_white:
        # Make near-white pixels transparent
        px = sub.load()
        w, hh = sub.size
        thresh = 255 - tolerance
        for y in range(hh):
            for x in range(w):
                r, g, b, a = px[x, y]
                if r >= thresh and g >= thresh and b >= thresh:
                    px[x, y] = (r, g, b, 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.save(str(out))
    print(json.dumps({
        "out": str(out), "size": sub.size,
        "transparent_white": transparent_white,
    }, indent=2))


# ---------------------------------------------------------------------------
# Measure — pixel<->inch helpers
# ---------------------------------------------------------------------------

def measure(img: Path, slide_w_in: float, slide_h_in: float) -> None:
    from PIL import Image
    im = Image.open(img)
    pw, ph = im.size
    print(json.dumps({
        "image_px": [pw, ph],
        "slide_in": [slide_w_in, slide_h_in],
        "px_per_in_x": pw / slide_w_in,
        "px_per_in_y": ph / slide_h_in,
        "inch_per_px_x": slide_w_in / pw,
        "inch_per_px_y": slide_h_in / ph,
        "hint": ("multiply pixel x by inch_per_px_x to get slide x; "
                  "use the same for y."),
    }, indent=2))


def pixelpick(img: Path, coords: list[tuple[int, int]]) -> None:
    from PIL import Image
    im = Image.open(img).convert("RGB")
    out = []
    for x, y in coords:
        r, g, b = im.getpixel((x, y))
        out.append({"px": [x, y], "hex": f"#{r:02X}{g:02X}{b:02X}",
                     "rgb": [r, g, b]})
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="sub", required=True)

    p = sub.add_parser("build", help="render JSON spec → pptx")
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("extract", help="rasterize PDF / copy PNG into work dir")
    p.add_argument("--src", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dpi", type=int, default=200)

    p = sub.add_parser("crop", help="crop a sub-region as a PNG asset")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--box", required=True,
                    help="left,top,right,bottom (px)")
    p.add_argument("--transparent-white", action="store_true",
                    help="make near-white pixels transparent")
    p.add_argument("--tolerance", type=int, default=8)

    p = sub.add_parser("measure", help="px<->in conversion table")
    p.add_argument("--src", required=True)
    p.add_argument("--width-in", type=float, default=13.333)
    p.add_argument("--height-in", type=float, default=7.5)

    p = sub.add_parser("pixelpick", help="dump colors at pixel coords")
    p.add_argument("--src", required=True)
    p.add_argument("--coords", required=True,
                    help="x1,y1;x2,y2;...")

    args = ap.parse_args(argv)
    if args.sub == "build":
        build(Path(args.spec), Path(args.out))
        return 0
    if args.sub == "extract":
        extract(Path(args.src), Path(args.out_dir), args.dpi)
        return 0
    if args.sub == "crop":
        box = tuple(int(v) for v in args.box.split(","))
        if len(box) != 4:
            raise SystemExit("--box must be left,top,right,bottom")
        crop(Path(args.src), Path(args.out), box,
              transparent_white=args.transparent_white,
              tolerance=args.tolerance)
        return 0
    if args.sub == "measure":
        measure(Path(args.src), args.width_in, args.height_in)
        return 0
    if args.sub == "pixelpick":
        coords = [tuple(int(v) for v in pt.split(","))
                   for pt in args.coords.split(";")]
        pixelpick(Path(args.src), coords)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
