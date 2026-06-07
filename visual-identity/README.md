# Vitali Set visual identity — generating post covers with AI

> Internal process docs. This whole `visual-identity/` folder is listed in `_config.yml > exclude`,
> so it is **not** published with the site.
>
> - `README.md` — this guide (style bible, Gemini workflow, prompt template, export).
> - `example-prompts.md` — ready-to-paste prompts, one per topic, to feed into Gemini.
> - `generations/` — suggested place to save the images you generate (create it as you go).

Goal: define a **new, internally consistent** visual identity for post covers (`featured-img`)
generated with AI, anchored on the blog's **chicken mascot** (the current logo). This is *not*
about reproducing the previous, painterly robot style by
[Lucas Álamo](https://www.instagram.com/lucasalamoart/) — that earlier identity is retired and
honored as an archive at `/gallery/`. From here on, consistency comes from the chicken + the
logo-derived palette + the "visual metaphor" idea, described below.

Chosen tool: **Google Gemini (the "Nano Banana" image model)** — available through
[Gemini](https://gemini.google.com/) or [Google AI Studio](https://aistudio.google.com/);
strong at keeping a character/style consistent from reference images and at conversational editing.

---

## 1. Style bible (the consistency anchors)

This is what should make a cover "look like the blog" going forward. Lock these down on every
generation:

- **Mascot — the chicken:** the same character as the logo (`assets/img/icons/favicon.svg`,
  `android-chrome-256x256.png`): a friendly chicken with a **white/cream face**, a **red comb**,
  a **golden-yellow beak**, **bold black outlines**, and simple round dot eyes. Keep it clearly
  recognizable as *that* chicken. It is **recurring but not mandatory** — it's the protagonist of
  most covers, but a purely conceptual/metaphorical cover without it is fine.
- **Technique — flat-with-depth:** start from the logo's **flat, bold black-outline** look, but
  add **light shading, subtle texture, and gentle gradients** for depth and life. A middle ground
  between a flat emoji and a fully painted illustration. **Avoid:** heavy painterly brushwork,
  photorealism, busy 3D renders, the glossy generic "AI art" look.
- **Palette — derived from the logo:** **red** (comb), **golden-yellow** (beak), **black**
  (outlines), and **white/cream** as the base, plus **1–2 consistent accent colors** of your
  choice. A strong, recognizable color identity is part of the brand.
- **Concept = visual metaphor** for the post's topic, not a literal illustration. (This principle
  carries over from the old identity because it works well — e.g. distance → the chicken measuring
  something with rulers; threshold → a conveyor belt of eggs being sorted "OK / not OK".)
- **Composition:** **widescreen 16:9** (covers render between ~1.7:1 and ~1.85:1), central subject,
  with breathing room at the top because the theme overlays the **title** on the image.

---

## 2. Gemini workflow (step by step)

1. **Start a fresh chat** in Gemini / AI Studio.
2. **Attach the reference image(s):** the logo chicken
   (`assets/img/icons/android-chrome-256x256.png`). As you produce covers you're happy with,
   keep your best one or two as **extra anchor references** and reuse them every time — that's
   what keeps the chicken consistent across posts over the long run.
3. **Send the prompt** (template below, or one from `example-prompts.md`), explicit: *"use the
   attached image as the character/style reference; same chicken and same look, new scene."*
4. **Iterate by conversational editing** instead of starting over:
   - "keep the same chicken and style, change the background to ..."
   - "thicker black outlines", "add a bit of soft shading", "stick to the red/yellow/cream palette".
5. **Aspect ratio 16:9.** If the tool outputs a square, explicitly ask for *widescreen 16:9*, or
   crop/extend afterward.
6. **Download at the highest resolution available.** Aim for **≥ 1600 px wide** for the base file.
7. **Check it against the style bible** (section 1) before adopting. If it drifts, go back to step 4.

---

## 3. Prompt template (fill in the `[...]`)

The **STYLE block is fixed** — paste it verbatim on every generation. Only change the first line
and the scene. See `example-prompts.md` for this template filled in for real post topics.

```
Concept: [the post's idea in one sentence].
Scene: the chicken mascot [specific action/scene, props, how the accents are used, mood].

STYLE: modern flat-design vector illustration that extends the blog's existing logo, keeping the
SAME chicken character. Confident, even bold black (#000000) outlines with uniform medium-thick
line weight and rounded caps/joins. Go beyond pure flat: clean cel-shading with one or two extra
shadow tones per color, soft smooth gradients, and a faint paper-grain texture so it feels crafted,
not sterile. Soft studio lighting from the upper-left with a gentle contact shadow under the subject.
Strict palette: comb red #EA5A47, golden-yellow #F1B31C, black #000000 outlines, white/cream
#FFFFFF–#FAF3E0 base, with teal #1FA8A0 and soft blue #5B9BD5 as the only accent colors.
Cohesive, friendly, polished editorial-illustration quality; crisp, sharp, high-resolution.

COMPOSITION: widescreen 16:9, chicken as a clear central focal point, simple uncluttered background,
generous negative space and a clean margin across the top third for a title overlay.

AVOID: heavy painterly oil brushwork, photorealism, realistic feather detail, 3D / Pixar render,
glossy plastic shading, neon cyberpunk look, cluttered busy backgrounds, muddy or off-palette
colors, any text / letters / numbers / words, watermarks, signatures, extra limbs, deformed beak.

[Attach the logo chicken + your master reference as character/style reference.]
```

The accent colors (teal/soft blue) are a choice — keep them the same across every cover for a
unified set. See `example-prompts.md` for this filled in across real topics, plus a "master
reference" prompt to run first.

> Note: the chicken is your own logo (derived from [OpenMoji](https://openmoji.org/), CC BY-SA 4.0,
> attributed in `README.md`). Keeping new covers recognizable as that same character is the point.

---

## 4. Exporting to the blog

The theme uses the `featured-img: <name>` front matter and resolves it to
`assets/img/posts/<name>.jpg`. For a new cover to work you only need **4 files** (the hero uses
the base; the home card uses thumb/placehold):

| file | use | suggested size |
|---|---|---|
| `<name>.jpg`            | hero (top of the post)     | ~1600 px wide |
| `<name>_thumb.jpg`      | card on the home page      | ~400 px wide |
| `<name>_thumb@2x.jpg`   | card on retina screens     | ~800 px wide |
| `<name>_placehold.jpg`  | blurred lazy-load placeholder | ~40 px, very small |

> The `_lg/_md/_sm/_xs` variants exist on the old covers but are **not** used by the current
> templates — you can generate them for completeness, but they aren't required.

Python (Pillow) snippet to generate the 4 variants from the base — run from the repo root:

```python
from PIL import Image, ImageFilter
from pathlib import Path

POSTS = Path("assets/img/posts")
name = "overfitting"          # <- the featured-img name
base = Image.open(POSTS / f"{name}.jpg").convert("RGB")

def save_w(img, width, suffix, blur=0):
    h = round(img.height * width / img.width)
    out = img.resize((width, h), Image.LANCZOS)
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    out.save(POSTS / f"{name}{suffix}.jpg", quality=85, optimize=True)

save_w(base, 400, "_thumb")
save_w(base, 800, "_thumb@2x")
save_w(base, 40,  "_placehold", blur=2)
print("ok")
```

Then just point the post at the new cover:

```yaml
featured-img: overfitting
```

And do **not** delete the old post images — they live on in `/gallery/`.
