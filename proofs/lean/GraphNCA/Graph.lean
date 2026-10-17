import Mathlib

set_option autoImplicit false

namespace GraphNCA

def graphStep {V R S : Type*} (neighbors : R → V → V)
    (rule : S → (R → S) → S) (x : V → S) (i : V) : S :=
  rule (x i) (fun r => x (neighbors r i))

def relabelState {V S : Type*} (σ : Equiv.Perm V) (x : V → S) : V → S :=
  fun i => x (σ.symm i)

def relabelNeighbors {V R : Type*} (σ : Equiv.Perm V)
    (neighbors : R → V → V) : R → V → V :=
  fun r i => σ (neighbors r (σ.symm i))

/-- Shared typed local rules commute with consistent vertex relabeling. -/
theorem graphStep_equivariant {V R S : Type*} (σ : Equiv.Perm V)
    (neighbors : R → V → V) (rule : S → (R → S) → S) (x : V → S) :
    graphStep (relabelNeighbors σ neighbors) rule (relabelState σ x) =
      relabelState σ (graphStep neighbors rule x) := by
  funext i
  simp [graphStep, relabelNeighbors, relabelState]

/-- Swap the receiver slots a and b of a single typed source permutation. -/
def swappedRelation {V : Type*} [DecidableEq V] (p : Equiv.Perm V)
    (a b : V) : Equiv.Perm V :=
  (Equiv.swap a b).trans p

theorem swappedRelation_bijective {V : Type*} [DecidableEq V]
    (p : Equiv.Perm V) (a b : V) :
    Function.Bijective (swappedRelation p a b) :=
  (swappedRelation p a b).bijective

/-- Every source still occurs at exactly one receiver for this relation. -/
theorem swappedRelation_unique_preimage {V : Type*} [DecidableEq V]
    (p : Equiv.Perm V) (a b source : V) :
    ∃ i, swappedRelation p a b i = source ∧
      ∀ j, swappedRelation p a b j = source → j = i := by
  refine ⟨(swappedRelation p a b).symm source, ?_, ?_⟩
  · exact (swappedRelation p a b).apply_symm_apply source
  · intro j hj
    apply (swappedRelation p a b).injective
    simpa using hj

open scoped BigOperators

/-- Total serialization lag is invariant under a typed source permutation. -/
theorem totalLag_permutation {V : Type*} [Fintype V]
    (σ : Equiv.Perm V) (patch : V → ℝ) (frameLength : ℝ) :
    (∑ i, (frameLength + patch i - patch (σ i))) =
      (Fintype.card V : ℝ) * frameLength := by
  simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib]
  rw [Equiv.sum_comp σ patch]
  simp

/-- With at least one vertex, uniform mean lag is exactly frameLength. -/
theorem meanLag_permutation {V : Type*} [Fintype V] [Nonempty V]
    (σ : Equiv.Perm V) (patch : V → ℝ) (frameLength : ℝ) :
    (∑ i, (frameLength + patch i - patch (σ i))) /
      (Fintype.card V : ℝ) = frameLength := by
  rw [totalLag_permutation]
  have hn : (Fintype.card V : ℝ) ≠ 0 := by
    exact_mod_cast (Fintype.card_ne_zero : Fintype.card V ≠ 0)
  field_simp

end GraphNCA
