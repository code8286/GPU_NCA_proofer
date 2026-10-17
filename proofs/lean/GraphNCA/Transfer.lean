import Mathlib

set_option autoImplicit false

namespace GraphNCA

def resetHead {B S T : Type*} (head : T) (parameters : B × S) : B × T :=
  (parameters.1, head)

theorem resetHead_preserves_backbone {B S T : Type*}
    (head : T) (parameters : B × S) :
    (resetHead head parameters).1 = parameters.1 := rfl

theorem resetHead_discards_synthetic_head {B S T : Type*}
    (head : T) (backbone : B) (first second : S) :
    resetHead head (backbone, first) = resetHead head (backbone, second) := rfl

theorem resetHead_idempotent {B S T : Type*} (head : T) (parameters : B × S) :
    resetHead head (resetHead head parameters) = resetHead head parameters := rfl

/-- Algebraic consequence of an ASSUMED quadratic Taylor upper bound. -/
theorem descent_from_remainder_bound (oldLoss newLoss linear curvature : ℝ)
    (hbound : newLoss - oldLoss ≤ linear + curvature)
    (halign : linear + curvature < 0) :
    newLoss < oldLoss := by linarith

/-- The analytic smoothness-to-remainder implication is not assumed proved here. -/
theorem sufficient_backbone_transfer {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] (f : E → ℝ) (x delta gradient : E) (L : ℝ)
    (hbound : f (x + delta) - f x ≤
      inner ℝ gradient delta + L / 2 * ‖delta‖ ^ 2)
    (halign : -inner ℝ gradient delta > L / 2 * ‖delta‖ ^ 2) :
    f (x + delta) < f x := by
  apply descent_from_remainder_bound (f x) (f (x + delta))
    (inner ℝ gradient delta) (L / 2 * ‖delta‖ ^ 2) hbound
  linarith

theorem one_step_alignment (eta alignment L squaredNorm : ℝ)
    (heta : 0 < eta) (h : L * eta / 2 * squaredNorm < alignment) :
    -eta * alignment + L / 2 * eta ^ 2 * squaredNorm < 0 := by
  nlinarith [mul_pos heta (sub_pos.mpr h)]

/-- Two-sided remainders are needed when comparing two treatments. -/
theorem compare_from_two_sided_bounds
    (graphLoss gridLoss base graphLinear gridLinear graphError gridError : ℝ)
    (hgraph : graphLoss - base ≤ graphLinear + graphError)
    (hgrid : gridLinear - gridError ≤ gridLoss - base) :
    graphLoss - gridLoss ≤ graphLinear - gridLinear + graphError + gridError := by
  linarith

theorem total_cost_break_even
    (baselineSteps graphSteps syntheticSteps generation langCost pptCost : ℝ) :
    generation + pptCost * syntheticSteps + langCost * graphSteps <
      langCost * baselineSteps ↔
    generation + pptCost * syntheticSteps <
      langCost * (baselineSteps - graphSteps) := by
  constructor <;> intro h <;> nlinarith

end GraphNCA
