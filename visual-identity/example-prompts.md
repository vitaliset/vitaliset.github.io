# Example prompts — chicken-mascot covers

Ready-to-paste prompts for **Google Gemini (Nano Banana)**, following the template in
[`README.md`](README.md). I can't render images here, so these are the prompts that *produce* the
covers — paste one into Gemini, **attach the logo chicken**
(`assets/img/icons/android-chrome-256x256.png`) as the character/style reference, and iterate.

How to use this file:
1. Read **THE STYLE BLOCK** below once — it's the fixed art direction, pasted into every prompt.
2. Start with the **master reference** prompt to lock the character design.
3. Then pick a topic prompt. Each one is **self-contained and paste-ready** (the style block is
   already included), so you can copy a whole code block straight into Gemini.

Conventions:
- Exact brand colors from the logo: comb red `#EA5A47`, golden-yellow beak `#F1B31C`, black
  outlines `#000000`, white/cream face `#FFFFFF`→`#FAF3E0`.
- Accents fixed to **teal `#1FA8A0` + soft blue `#5B9BD5`** across every prompt, so the whole set
  stays chromatically unified. Change them if you want — just keep them the same everywhere.
- Save outputs into `visual-identity/generations/`, and once happy, reuse your best one as an extra
  reference image to keep the chicken from drifting.

---

## THE STYLE BLOCK (the fixed art direction)

This is the part that makes everything look like one coherent set. It's baked into every prompt
below; this copy is here for reference and for writing new prompts.

```
STYLE: modern flat-design vector illustration that extends the blog's existing logo, keeping the
SAME chicken character. Confident, even bold black (#000000) outlines with uniform medium-thick
line weight and rounded caps/joins. Go beyond pure flat: clean cel-shading with one or two extra
shadow tones per color, soft smooth gradients, and a faint paper-grain texture so it feels crafted,
not sterile. Soft studio lighting from the upper-left with a gentle contact shadow under the subject.
Strict palette: comb red #EA5A47, golden-yellow #F1B31C, black #000000 outlines, white/cream
#FFFFFF–#FAF3E0 base, with teal #1FA8A0 and soft blue #5B9BD5 as the only accent colors.
Cohesive, friendly, polished editorial-illustration quality; crisp, sharp, high-resolution.

COMPOSITION: widescreen 16:9. The chicken is a clear central focal point at comfortable scale; the
background is simple and uncluttered (flat color or a very subtle gradient with a few minimal
shapes); keep generous negative space and a clean, calm margin across the top third for a title
overlay. Balanced, poster-like.

AVOID: heavy painterly oil brushwork, photorealism, realistic feather detail, 3D / Pixar render,
glossy plastic shading, neon cyberpunk look, cluttered busy backgrounds, muddy or off-palette
colors, any text / letters / numbers / words, watermarks, signatures, extra limbs, deformed beak,
malformed eyes.
```

---

## 0. Master reference (do this FIRST)

A single clean full-body character design. The logo only shows the face, so this establishes the
whole chicken — attach it to every later prompt and the character stops drifting.

```
A character design sheet of a friendly cartoon chicken mascot — the SAME character as the attached
logo: rounded white/cream egg-shaped body and head, a small bright red (#EA5A47) comb and wattle, a
short golden-yellow (#F1B31C) triangular beak, two simple round black dot eyes, little stubby wings,
two thin orange legs with three-toed feet. Standing in a neutral friendly three-quarter pose,
plus a smaller secondary view of the face close-up beside it. Approachable, confident, a little
witty — a mascot you'd trust to explain machine learning.

STYLE: modern flat-design vector illustration that extends the blog's existing logo, keeping the
SAME chicken character. Confident, even bold black (#000000) outlines with uniform medium-thick
line weight and rounded caps/joins. Go beyond pure flat: clean cel-shading with one or two extra
shadow tones per color, soft smooth gradients, and a faint paper-grain texture so it feels crafted,
not sterile. Soft studio lighting from the upper-left with a gentle contact shadow under the subject.
Strict palette: comb red #EA5A47, golden-yellow #F1B31C, black #000000 outlines, white/cream
#FFFFFF–#FAF3E0 base, with teal #1FA8A0 and soft blue #5B9BD5 as the only accent colors.
Cohesive, friendly, polished editorial-illustration quality; crisp, sharp, high-resolution.

COMPOSITION: widescreen 16:9, centered, on a plain off-white background, lots of clean space.

AVOID: heavy painterly oil brushwork, photorealism, realistic feather detail, 3D / Pixar render,
glossy plastic shading, neon cyberpunk look, cluttered busy backgrounds, muddy or off-palette
colors, any text / letters / numbers / words, watermarks, signatures, extra limbs, deformed beak,
malformed eyes.

[Attach the logo chicken as the character reference.]
```

> Once you have a master you like, **save it** and attach it (together with the logo) to every prompt
> below, adding the line: *"keep the exact same chicken character as the attached reference."*

---

## 1. Generalizando distância → `coverdistancia`
*Metaphor: there are many valid notions of "distance" between two points.*

```
Concept: there is more than one valid way to measure the distance between two things.
Scene: the chicken mascot standing between two glowing teal dots, stretching a long golden-yellow
measuring tape from one to the other, head tilted thoughtfully; faintly behind it, three different
"unit ball" shapes (a circle, a diamond and a rounded square in soft blue) hint at different
distance metrics. Calm, curious mood.

STYLE: modern flat-design vector illustration that extends the blog's existing logo, keeping the
SAME chicken character. Confident, even bold black (#000000) outlines with uniform medium-thick
line weight and rounded caps/joins. Clean cel-shading with one or two extra shadow tones per color,
soft smooth gradients, and a faint paper-grain texture. Soft studio lighting from the upper-left,
gentle contact shadow. Strict palette: comb red #EA5A47, golden-yellow #F1B31C, black #000000
outlines, white/cream #FFFFFF–#FAF3E0 base, with teal #1FA8A0 and soft blue #5B9BD5 accents.
Cohesive, friendly, polished editorial-illustration quality; crisp, sharp, high-resolution.

COMPOSITION: widescreen 16:9, chicken centered, uncluttered background, clean margin across the top
third for a title overlay.

AVOID: heavy painterly oil brushwork, photorealism, realistic feather detail, 3D / Pixar render,
glossy plastic shading, neon cyberpunk look, cluttered busy backgrounds, muddy or off-palette
colors, any text / letters / numbers / words, watermarks, signatures, extra limbs, deformed beak.

[Attach the logo chicken + your master reference. Keep the exact same chicken character.]
```

---

## 2. Motivando k-Fold → `coverkfold`
*Metaphor: split the data into k folds, rotating which one is held out for testing.*

```
Concept: cross-validation splits the data into k folds and rotates which fold is held out.
Scene: the chicken mascot using a wing to divide a long horizontal strip of grain/seeds into five
equal numbered segments; four segments glow soft blue ("training") and one glows teal ("held-out
test"), clearly set apart. The chicken looks focused and tidy, like it's organizing the strip.

STYLE: modern flat-design vector illustration that extends the blog's existing logo, keeping the
SAME chicken character. Confident, even bold black (#000000) outlines with uniform medium-thick
line weight and rounded caps/joins. Clean cel-shading with one or two extra shadow tones per color,
soft smooth gradients, and a faint paper-grain texture. Soft studio lighting from the upper-left,
gentle contact shadow. Strict palette: comb red #EA5A47, golden-yellow #F1B31C, black #000000
outlines, white/cream #FFFFFF–#FAF3E0 base, with teal #1FA8A0 and soft blue #5B9BD5 accents.
Cohesive, friendly, polished editorial-illustration quality; crisp, sharp, high-resolution.

COMPOSITION: widescreen 16:9, chicken centered, uncluttered background, clean margin across the top
third for a title overlay.

AVOID: heavy painterly oil brushwork, photorealism, realistic feather detail, 3D / Pixar render,
glossy plastic shading, neon cyberpunk look, cluttered busy backgrounds, muddy or off-palette
colors, any text / letters / numbers / words, watermarks, signatures, extra limbs, deformed beak.

[Attach the logo chicken + your master reference. Keep the exact same chicken character.]
```

---

## 3. Covariate Shift: Introduction → `covariate_0_formulando`
*Metaphor: the input distribution at test time differs from training.*

```
Concept: the input data distribution shifts between training and deployment.
Scene: a split landscape — on the left a calm soft-blue meadow where seeds are scattered evenly, on
the right a teal meadow where the same seeds are clumped together differently. The chicken mascot is
mid-step across the dividing line, one foot on each side, glancing back with a slightly puzzled
expression at how the ground has changed under it.

STYLE: modern flat-design vector illustration that extends the blog's existing logo, keeping the
SAME chicken character. Confident, even bold black (#000000) outlines with uniform medium-thick
line weight and rounded caps/joins. Clean cel-shading with one or two extra shadow tones per color,
soft smooth gradients, and a faint paper-grain texture. Soft studio lighting from the upper-left,
gentle contact shadow. Strict palette: comb red #EA5A47, golden-yellow #F1B31C, black #000000
outlines, white/cream #FFFFFF–#FAF3E0 base, with teal #1FA8A0 and soft blue #5B9BD5 accents.
Cohesive, friendly, polished editorial-illustration quality; crisp, sharp, high-resolution.

COMPOSITION: widescreen 16:9, chicken centered on the dividing line, uncluttered background, clean
margin across the top third for a title overlay.

AVOID: heavy painterly oil brushwork, photorealism, realistic feather detail, 3D / Pixar render,
glossy plastic shading, neon cyberpunk look, cluttered busy backgrounds, muddy or off-palette
colors, any text / letters / numbers / words, watermarks, signatures, extra limbs, deformed beak.

[Attach the logo chicken + your master reference. Keep the exact same chicken character.]
```

---

## 4. Uma utilização crítica do Boruta → `boruta`
*Metaphor: keep features only if they beat randomized "shadow" copies.*

```
Concept: a feature is kept only if it outperforms its randomized "shadow" copy.
Scene: the chicken mascot at a tidy desk, sorting seeds into two trays — bright golden-yellow
"real" seeds it keeps, and dull grey translucent "shadow" seeds it flicks away into a small bin.
A soft-blue balance scale on the desk tips in favor of the golden seeds. Discerning, slightly
skeptical expression.

STYLE: modern flat-design vector illustration that extends the blog's existing logo, keeping the
SAME chicken character. Confident, even bold black (#000000) outlines with uniform medium-thick
line weight and rounded caps/joins. Clean cel-shading with one or two extra shadow tones per color,
soft smooth gradients, and a faint paper-grain texture. Soft studio lighting from the upper-left,
gentle contact shadow. Strict palette: comb red #EA5A47, golden-yellow #F1B31C, black #000000
outlines, white/cream #FFFFFF–#FAF3E0 base, with teal #1FA8A0 and soft blue #5B9BD5 accents.
Cohesive, friendly, polished editorial-illustration quality; crisp, sharp, high-resolution.

COMPOSITION: widescreen 16:9, chicken centered, uncluttered background, clean margin across the top
third for a title overlay.

AVOID: heavy painterly oil brushwork, photorealism, realistic feather detail, 3D / Pixar render,
glossy plastic shading, neon cyberpunk look, cluttered busy backgrounds, muddy or off-palette
colors, any text / letters / numbers / words, watermarks, signatures, extra limbs, deformed beak.

[Attach the logo chicken + your master reference. Keep the exact same chicken character.]
```

---

## 5. Meta K-Means: um ensemble de K-Means → `metakmeans`
*Metaphor: combine many clusterings into one consensus grouping.*

```
Concept: combine several different clusterings into one consensus grouping.
Scene: the chicken mascot as a calm shepherd herding scattered little fluffy chicks into three
tidy, clearly separated flocks — one circled in teal, one in soft blue, one in golden-yellow. A few
stray chicks are being gently nudged toward the right group. Orderly, satisfied mood.

STYLE: modern flat-design vector illustration that extends the blog's existing logo, keeping the
SAME chicken character. Confident, even bold black (#000000) outlines with uniform medium-thick
line weight and rounded caps/joins. Clean cel-shading with one or two extra shadow tones per color,
soft smooth gradients, and a faint paper-grain texture. Soft studio lighting from the upper-left,
gentle contact shadow. Strict palette: comb red #EA5A47, golden-yellow #F1B31C, black #000000
outlines, white/cream #FFFFFF–#FAF3E0 base, with teal #1FA8A0 and soft blue #5B9BD5 accents.
Cohesive, friendly, polished editorial-illustration quality; crisp, sharp, high-resolution.

COMPOSITION: widescreen 16:9, chicken centered among the flocks, uncluttered background, clean
margin across the top third for a title overlay.

AVOID: heavy painterly oil brushwork, photorealism, realistic feather detail, 3D / Pixar render,
glossy plastic shading, neon cyberpunk look, cluttered busy backgrounds, muddy or off-palette
colors, any text / letters / numbers / words, watermarks, signatures, extra limbs, deformed beak.

[Attach the logo chicken + your master reference. Keep the exact same chicken character.]
```

---

## 6. Threshold-dependent metrics → `threshold_dependent_opt`
*Metaphor: the decision threshold changes which items are accepted, and the metric with it.*

```
Concept: tuning a decision threshold changes which items get accepted, and the metric along with it.
Scene: the chicken mascot turning a large golden-yellow dial beside a conveyor belt of eggs; an
adjustable gate splits the eggs into a teal "accept" bin and a soft-blue "reject" bin, and the
split visibly shifts with the dial. Focused, hands-on expression, like an operator at a control.

STYLE: modern flat-design vector illustration that extends the blog's existing logo, keeping the
SAME chicken character. Confident, even bold black (#000000) outlines with uniform medium-thick
line weight and rounded caps/joins. Clean cel-shading with one or two extra shadow tones per color,
soft smooth gradients, and a faint paper-grain texture. Soft studio lighting from the upper-left,
gentle contact shadow. Strict palette: comb red #EA5A47, golden-yellow #F1B31C, black #000000
outlines, white/cream #FFFFFF–#FAF3E0 base, with teal #1FA8A0 and soft blue #5B9BD5 accents.
Cohesive, friendly, polished editorial-illustration quality; crisp, sharp, high-resolution.

COMPOSITION: widescreen 16:9, chicken and dial as the focal point, uncluttered background, clean
margin across the top third for a title overlay.

AVOID: heavy painterly oil brushwork, photorealism, realistic feather detail, 3D / Pixar render,
glossy plastic shading, neon cyberpunk look, cluttered busy backgrounds, muddy or off-palette
colors, any text / letters / numbers / words, watermarks, signatures, extra limbs, deformed beak.

[Attach the logo chicken + your master reference. Keep the exact same chicken character.]
```

---

## 7. Conditional Density Estimation → `cde`
*Metaphor: predict a whole distribution of outcomes, not a single value.*

```
Concept: predict an entire distribution of possible outcomes instead of one single value.
Scene: the chicken mascot looking up in soft wonder at a fanned-out spread of translucent eggs that
form a smooth bell-shaped probability cloud above its head, the eggs densest in the middle and
sparse at the edges, glowing in teal and soft blue. Gentle, contemplative mood.

STYLE: modern flat-design vector illustration that extends the blog's existing logo, keeping the
SAME chicken character. Confident, even bold black (#000000) outlines with uniform medium-thick
line weight and rounded caps/joins. Clean cel-shading with one or two extra shadow tones per color,
soft smooth gradients, and a faint paper-grain texture. Soft studio lighting from the upper-left,
gentle contact shadow. Strict palette: comb red #EA5A47, golden-yellow #F1B31C, black #000000
outlines, white/cream #FFFFFF–#FAF3E0 base, with teal #1FA8A0 and soft blue #5B9BD5 accents.
Cohesive, friendly, polished editorial-illustration quality; crisp, sharp, high-resolution.

COMPOSITION: widescreen 16:9, chicken low-center with the probability cloud filling the upper area,
uncluttered background, clean margin across the top third for a title overlay.

AVOID: heavy painterly oil brushwork, photorealism, realistic feather detail, 3D / Pixar render,
glossy plastic shading, neon cyberpunk look, cluttered busy backgrounds, muddy or off-palette
colors, any text / letters / numbers / words, watermarks, signatures, extra limbs, deformed beak.

[Attach the logo chicken + your master reference. Keep the exact same chicken character.]
```

---

## 8. The R² score does not vary between 0 and 1 → `r_squared`
*Metaphor: R² can dip below zero — the 0-to-1 intuition is wrong.*

```
Concept: the R-squared score can fall below zero, breaking the common 0-to-1 intuition.
Scene: the chicken mascot staring, eyebrows up in surprise, at a large round gauge whose needle has
swung left past the 0 mark into a red (#EA5A47) "negative" zone; the safe part of the dial is teal
and soft blue. A small puff of confusion above the chicken's head. Comic, mildly alarmed mood.

STYLE: modern flat-design vector illustration that extends the blog's existing logo, keeping the
SAME chicken character. Confident, even bold black (#000000) outlines with uniform medium-thick
line weight and rounded caps/joins. Clean cel-shading with one or two extra shadow tones per color,
soft smooth gradients, and a faint paper-grain texture. Soft studio lighting from the upper-left,
gentle contact shadow. Strict palette: comb red #EA5A47, golden-yellow #F1B31C, black #000000
outlines, white/cream #FFFFFF–#FAF3E0 base, with teal #1FA8A0 and soft blue #5B9BD5 accents.
Cohesive, friendly, polished editorial-illustration quality; crisp, sharp, high-resolution.

COMPOSITION: widescreen 16:9, chicken and gauge as the focal point, uncluttered background, clean
margin across the top third for a title overlay.

AVOID: heavy painterly oil brushwork, photorealism, realistic feather detail, 3D / Pixar render,
glossy plastic shading, neon cyberpunk look, cluttered busy backgrounds, muddy or off-palette
colors, any text / letters / numbers / words, watermarks, signatures, extra limbs, deformed beak.

[Attach the logo chicken + your master reference. Keep the exact same chicken character.]
```

---

## More metaphor seeds (for the remaining / future posts)

Same recipe — write a `Concept:` line and a `Scene:` line, then paste THE STYLE BLOCK and the
COMPOSITION / AVOID lines:

| Post (`featured-img`) | One-line metaphor | Scene idea |
|---|---|---|
| Covariate Shift: QQ-plot (`coverqqplot`) | comparing two distributions point-by-point | chicken matching seeds from two piles along a diagonal teal line |
| Covariate Shift: Classificador (`coverclassificador_binario`) | a classifier telling train from test data | chicken as a friendly bouncer waving samples into "train" / "test" doors |
| Conformal prediction in CATE (`cqr_cate`) | prediction *intervals* with coverage guarantees | chicken holding a soft-blue umbrella-shaped confidence band over a row of eggs |
| Evaluating ranking in regression (`ranking_regression`) | the *order* matters, not the exact values | chicken lining eggs up smallest-to-largest, checking the order is right |
| (future) overfitting | memorizing noise instead of the trend | chicken frantically pecking at every scattered seed instead of following the trend line |
```
