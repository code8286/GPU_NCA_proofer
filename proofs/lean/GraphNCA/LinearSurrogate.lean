import Mathlib

set_option autoImplicit false
open scoped BigOperators

namespace GraphNCA

def squaredSize {I : Type*} [Fintype I] (a : I → ℝ) : ℝ :=
  ∑ i, a i ^ 2

def crossInner {I : Type*} [Fintype I] (a c : I → ℝ) : ℝ :=
  ∑ i, a i * c i

noncomputable def squaredLoss {I : Type*} [Fintype I] (a c : I → ℝ) : ℝ :=
  ∑ i, (a i - c i) ^ 2 / 2

/-- Finite coordinates can be flattened matrix entries. -/
theorem linear_surrogate_expansion {I : Type*} [Fintype I]
    (gamma : ℝ) (a c : I → ℝ) :
    squaredLoss (fun i => gamma * a i) c - squaredLoss (fun _ => 0) c =
      gamma ^ 2 / 2 * squaredSize a - gamma * crossInner a c := by
  unfold squaredLoss
  rw [← Finset.sum_sub_distrib]
  have hcell : ∀ i, (gamma * a i - c i) ^ 2 / 2 - (0 - c i) ^ 2 / 2 =
      gamma ^ 2 / 2 * a i ^ 2 - gamma * (a i * c i) := by
    intro i
    ring
  simp_rw [hcell]
  simp [squaredSize, crossInner, Finset.sum_sub_distrib, ← Finset.mul_sum]

/-- Equal squared size is explicit; regular-degree counting is a separate obligation. -/
theorem equal_size_target_comparison {I : Type*} [Fintype I]
    (gamma : ℝ) (a b c : I → ℝ) (hsize : squaredSize a = squaredSize b) :
    squaredLoss (fun i => gamma * a i) c -
      squaredLoss (fun i => gamma * b i) c =
      -gamma * (crossInner a c - crossInner b c) := by
  have ha := linear_surrogate_expansion gamma a c
  have hb := linear_surrogate_expansion gamma b c
  rw [hsize] at ha
  linarith

theorem squared_difference_identity {I : Type*} [Fintype I] (a b : I → ℝ) :
    squaredSize (fun i => a i - b i) =
      squaredSize a + squaredSize b - 2 * crossInner a b := by
  unfold squaredSize crossInner
  calc
    (∑ i, (a i - b i) ^ 2) =
      ∑ i, (a i ^ 2 + b i ^ 2 - 2 * (a i * b i)) := by
        apply Finset.sum_congr rfl
        intro i _
        ring
    _ = _ := by
      simp [Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.mul_sum]

theorem synthetic_linear_recurrence
    (eta target : ℝ) (sequence : ℕ → ℝ) (hzero : sequence 0 = 0)
    (hstep : ∀ n, sequence (n + 1) = (1 - eta) * sequence n + eta * target) :
    ∀ n, sequence n = (1 - (1 - eta) ^ n) * target := by
  intro n
  induction n with
  | zero => simpa using hzero
  | succ n ih =>
    rw [hstep, ih, pow_succ]
    ring

end GraphNCA
