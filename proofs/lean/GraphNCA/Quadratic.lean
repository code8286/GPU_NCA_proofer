import Mathlib

set_option autoImplicit false
open scoped BigOperators

namespace GraphNCA

noncomputable def quadratic2 (lambda1 lambda2 x y : ℝ) : ℝ :=
  (lambda1 * x ^ 2 + lambda2 * y ^ 2) / 2

theorem scalar_recurrence_solution
    (factor initial : ℝ) (sequence : ℕ → ℝ) (hzero : sequence 0 = initial)
    (hstep : ∀ n, sequence (n + 1) = factor * sequence n) :
    ∀ n, sequence n = factor ^ n * initial := by
  intro n
  induction n with
  | zero => simpa using hzero
  | succ n ih =>
    rw [hstep, ih, pow_succ]
    ring

theorem two_mode_closed_form
    (lambda1 lambda2 r1 r2 c1 c2 : ℝ) (n : ℕ) :
    quadratic2 lambda1 lambda2 (r1 ^ n * c1) (r2 ^ n * c2) =
      (lambda1 * r1 ^ (2 * n) * c1 ^ 2 +
       lambda2 * r2 ^ (2 * n) * c2 ^ 2) / 2 := by
  have h1 : (r1 ^ n) ^ 2 = r1 ^ (2 * n) := by
    rw [← pow_mul, Nat.mul_comm n 2]
  have h2 : (r2 ^ n) ^ 2 = r2 ^ (2 * n) := by
    rw [← pow_mul, Nat.mul_comm n 2]
  simp only [quadratic2, mul_pow, h1, h2]
  ring

noncomputable def diagonalLoss {I : Type*} [Fintype I]
    (lambda factor coefficient : I → ℝ) (n : ℕ) : ℝ :=
  (∑ i, lambda i * factor i ^ (2 * n) * coefficient i ^ 2) / 2

theorem coefficient_dominance {I : Type*} [Fintype I]
    (lambda factor graph scratch : I → ℝ) (n : ℕ)
    (hlambda : ∀ i, 0 ≤ lambda i) (hfactor : ∀ i, 0 ≤ factor i)
    (hcoeff : ∀ i, graph i ^ 2 ≤ scratch i ^ 2) :
    diagonalLoss lambda factor graph n ≤ diagonalLoss lambda factor scratch n := by
  unfold diagonalLoss
  apply div_le_div_of_nonneg_right _ (by norm_num)
  apply Finset.sum_le_sum
  intro i _
  exact mul_le_mul_of_nonneg_left (hcoeff i)
    (mul_nonneg (hlambda i) (pow_nonneg (hfactor i) _))

theorem initial_loss_ranking :
    quadratic2 1 100 2 0 < quadratic2 1 100 1 (1 / 5) := by
  norm_num [quadratic2]

theorem after_one_step_ranking_reverses :
    quadratic2 1 100 ((1 - 1 / 100) * 1) ((1 - 100 * (1 / 100)) * (1 / 5)) <
      quadratic2 1 100 ((1 - 1 / 100) * 2) ((1 - 100 * (1 / 100)) * 0) := by
  norm_num [quadratic2]

end GraphNCA
