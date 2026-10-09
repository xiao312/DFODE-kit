# Chemistry surrogate research

This glossary defines the language used to compare learned chemistry increments.

## Language

**Target representation**: A reversible coordinate system for the chemistry increment that a model learns. It is not the complete training method.
_Avoid_: Method, model architecture

**Method recipe**: The specified input coordinates, target representation, approximator, loss, and optimization procedure used in an experiment.
_Avoid_: Representation when other factors also change

**Run**: One execution of a method recipe with a specified dataset, split, seed, precision, and computational budget.
_Avoid_: A new method merely because its seed changes

**Development split**: Chemistry states excluded from model fitting but inspected to guide research choices. Repeated inspection prevents it from being an untouched test.
_Avoid_: Independent test, unseen-case validation

**Independent-case test**: States from a separate physical case reserved until the method and acceptance criteria are fixed.
_Avoid_: Development snapshots from the same flame

**Increment acceptance**: A prediction error that meets a declared absolute-plus-relative budget based on the reference increment.
_Avoid_: Solver-level accuracy

**State-scaled acceptance**: A prediction error measured against a budget based on the reference state magnitude, not the increment magnitude. It answers a different question from increment acceptance.
_Avoid_: Increment acceptance, CVODE certification

**Tolerance parameters**: The absolute allowance parameter `atol` and dimensionless relative parameter `rtol` used to construct an error budget. The product `rtol * magnitude` is a contribution to that budget, not `rtol` itself.
_Avoid_: Calling the full allowed error `atol` or `rtol`

**Error-budget scaling**: Division of a physical prediction error by its declared allowed-error scale. It is distinct from encoding or standardizing the model's training target.
_Avoid_: Target representation, solver tolerance setting

**Component acceptance**: The fraction of scored species-state pairs that meet the declared error budget.
_Avoid_: Complete-state acceptance

**Complete-state acceptance**: The fraction of states for which every scored species meets the declared error budget.
_Avoid_: Mean species accuracy

**Reference qualification**: Evidence that estimated reference uncertainty is small enough for the requested error budget. An unqualified label is unknown, not a passing prediction.
_Avoid_: A guarantee from tighter solver tolerances alone
