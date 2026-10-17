import Mathlib

set_option autoImplicit false

namespace GraphNCA

def patchLag (patches target source : ℕ) : ℕ :=
  patches + 2 + target - source

theorem patchLag_bounds (patches target source : ℕ)
    (ht : target < patches) (hs : source < patches) :
    3 ≤ patchLag patches target source ∧
      patchLag patches target source ≤ 2 * patches + 1 := by
  unfold patchLag
  omega

theorem fullWindow_visibility (patches target source window : ℕ)
    (ht : target < patches) (hs : source < patches)
    (hw : 2 * patches + 1 ≤ window) :
    patchLag patches target source ≤ window :=
  le_trans (patchLag_bounds patches target source ht hs).2 hw

/-- Prefix availability is a separate parameter from configured window size. -/
theorem prefix_visibility (target source available : ℕ)
    (ht : target < 16) (hs : source < 16) (havailable : 33 ≤ available) :
    patchLag 16 target source ≤ available := by
  exact fullWindow_visibility 16 target source available ht hs havailable

theorem binary_parent_lag_fits (target source : ℕ)
    (ht : target < 16) (hs : source < 16) :
    patchLag 16 target source ≤ 128 := by
  exact fullWindow_visibility 16 target source 128 ht hs (by decide)

theorem crop_prefix_can_miss_parent : 32 < patchLag 16 15 0 := by decide

def encodeBinaryPatch (a b c d : Fin 2) : ℕ :=
  a.val + 2 * b.val + 4 * c.val + 8 * d.val

theorem encodeBinaryPatch_lt_sixteen (a b c d : Fin 2) :
    encodeBinaryPatch a b c d < 16 := by
  have ha := a.isLt
  have hb := b.isLt
  have hc := c.isLt
  have hd := d.isLt
  unfold encodeBinaryPatch
  omega

theorem encodeBinaryPatch_injective (a b c d e f g h : Fin 2)
    (heq : encodeBinaryPatch a b c d = encodeBinaryPatch e f g h) :
    a = e ∧ b = f ∧ c = g ∧ d = h := by
  have ha := a.isLt
  have hb := b.isLt
  have hc := c.isLt
  have hd := d.isLt
  have he := e.isLt
  have hf := f.isLt
  have hg := g.isLt
  have hh := h.isLt
  unfold encodeBinaryPatch at heq
  have h0 : a.val = e.val := by omega
  have h1 : b.val = f.val := by omega
  have h2 : c.val = g.val := by omega
  have h3 : d.val = h.val := by omega
  exact ⟨Fin.ext h0, Fin.ext h1, Fin.ext h2, Fin.ext h3⟩

theorem binary_patch_digits (a b c d : Fin 2) :
    encodeBinaryPatch a b c d % 2 = a.val ∧
    (encodeBinaryPatch a b c d / 2) % 2 = b.val ∧
    (encodeBinaryPatch a b c d / 4) % 2 = c.val ∧
    (encodeBinaryPatch a b c d / 8) % 2 = d.val := by
  have ha := a.isLt
  have hb := b.isLt
  have hc := c.isLt
  have hd := d.isLt
  unfold encodeBinaryPatch
  omega

end GraphNCA
