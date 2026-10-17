import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace GraphNCA

def scratchToyLoss (n : ℕ) : ℚ :=
  ((19 / 20 : ℚ) ^ (2 * n) + 10 * (1 / 2 : ℚ) ^ (2 * n)) / 2

def transferredToyLoss (n : ℕ) : ℚ :=
  ((1 / 4 : ℚ) * (19 / 20 : ℚ) ^ (2 * n) +
    10 * (1 / 2 : ℚ) ^ (2 * n)) / 2

theorem scratchToyLoss_at_38 : (1 / 100 : ℚ) < scratchToyLoss 38 := by
  norm_num [scratchToyLoss]

theorem scratchToyLoss_at_39 : scratchToyLoss 39 < (1 / 100 : ℚ) := by
  norm_num [scratchToyLoss]

theorem transferredToyLoss_at_24 : (1 / 100 : ℚ) < transferredToyLoss 24 := by
  norm_num [transferredToyLoss]

theorem transferredToyLoss_at_25 : transferredToyLoss 25 < (1 / 100 : ℚ) := by
  norm_num [transferredToyLoss]

theorem unit_interval_power_antitone (r : ℚ) (hzero : 0 ≤ r) (hone : r ≤ 1) :
    Antitone (fun n : ℕ => r ^ n) := by
  apply antitone_nat_of_succ_le
  intro n
  rw [pow_succ]
  exact mul_le_of_le_one_right (pow_nonneg hzero n) hone

theorem scratchToyLoss_antitone : Antitone scratchToyLoss := by
  intro n m hnm
  have hslow := unit_interval_power_antitone (19 / 20) (by norm_num) (by norm_num)
    (show 2 * n ≤ 2 * m by omega)
  have hfast := unit_interval_power_antitone (1 / 2) (by norm_num) (by norm_num)
    (show 2 * n ≤ 2 * m by omega)
  unfold scratchToyLoss
  linarith

theorem transferredToyLoss_antitone : Antitone transferredToyLoss := by
  intro n m hnm
  have hslow := unit_interval_power_antitone (19 / 20) (by norm_num) (by norm_num)
    (show 2 * n ≤ 2 * m by omega)
  have hfast := unit_interval_power_antitone (1 / 2) (by norm_num) (by norm_num)
    (show 2 * n ≤ 2 * m by omega)
  unfold transferredToyLoss
  linarith

theorem scratch_first_hit :
    scratchToyLoss 39 ≤ (1 / 100 : ℚ) ∧
      ∀ n : ℕ, n < 39 → (1 / 100 : ℚ) < scratchToyLoss n := by
  refine ⟨le_of_lt scratchToyLoss_at_39, ?_⟩
  intro n hn
  have hle : n ≤ 38 := by omega
  exact lt_of_lt_of_le scratchToyLoss_at_38 (scratchToyLoss_antitone hle)

theorem transferred_first_hit :
    transferredToyLoss 25 ≤ (1 / 100 : ℚ) ∧
      ∀ n : ℕ, n < 25 → (1 / 100 : ℚ) < transferredToyLoss n := by
  refine ⟨le_of_lt transferredToyLoss_at_25, ?_⟩
  intro n hn
  have hle : n ≤ 24 := by omega
  exact lt_of_lt_of_le transferredToyLoss_at_24 (transferredToyLoss_antitone hle)

theorem model_parameter_count :
    274 * 192 + 128 * 192 + 4 * (12 * 192 ^ 2 + 13 * 192) + 2 * 192 =
      (1857024 : ℕ) := by norm_num

theorem retained_token_count : 16 * (16 + 2) = (288 : ℕ) := by norm_num

theorem qualifying_prefix_count : 128 - 33 + 1 = (96 : ℕ) := by norm_num

theorem cost_examples :
    (39 / 25 : ℚ) = 156 / 100 ∧
    1 < (39 : ℚ) / (10 + 25) ∧
    (39 : ℚ) / (20 + 25) < 1 := by norm_num

theorem linear_surrogate_losses :
    (1 / 2 : ℚ) * 3 = 3 / 2 ∧
    (1 / 2 : ℚ) * (1 / 4) * 3 = 3 / 8 ∧
    (1 / 2 : ℚ) * ((1 / 4) * 3 + 3) = 15 / 8 := by norm_num

end GraphNCA
