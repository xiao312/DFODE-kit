from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import nn


OutOfBracketMode = Literal["raise", "clamp", "nan"]


class ThermochemicalClosureError(ValueError):
    """Base error for invalid thermochemical closure inputs or results."""


class NonPhysicalStateError(ThermochemicalClosureError):
    """Raised when a supplied pressure or composition is nonphysical."""


class TemperatureBracketError(ThermochemicalClosureError):
    """Raised when the target enthalpy is outside the temperature bracket."""


class TemperatureConvergenceError(ThermochemicalClosureError):
    """Raised when the safeguarded temperature solve does not converge."""


@dataclass(frozen=True)
class Nasa7ThermoData:
    """Portable ideal-gas NASA7 data for a fixed species ordering.

    Molecular weights use kg/kmol, temperatures use K, and ``gas_constant``
    uses J/(kmol K). The resulting species and mixture enthalpies use J/kg.
    The explicit mapping format is suitable for JSON manifests and model
    artifacts and does not require Cantera at inference time.
    """

    species_names: tuple[str, ...]
    molecular_weights: np.ndarray
    temperature_ranges: np.ndarray
    low_coefficients: np.ndarray
    high_coefficients: np.ndarray
    gas_constant: float = 8314.46261815324

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.species_names)
        molecular_weights = np.ascontiguousarray(self.molecular_weights, dtype=np.float64)
        temperature_ranges = np.ascontiguousarray(self.temperature_ranges, dtype=np.float64)
        low_coefficients = np.ascontiguousarray(self.low_coefficients, dtype=np.float64)
        high_coefficients = np.ascontiguousarray(self.high_coefficients, dtype=np.float64)
        n_species = len(names)

        if n_species == 0:
            raise ValueError("NASA7 data must contain at least one species")
        if molecular_weights.shape != (n_species,):
            raise ValueError("molecular_weights must have shape (n_species,)")
        if temperature_ranges.shape != (n_species, 3):
            raise ValueError("temperature_ranges must have shape (n_species, 3)")
        if low_coefficients.shape != (n_species, 7):
            raise ValueError("low_coefficients must have shape (n_species, 7)")
        if high_coefficients.shape != (n_species, 7):
            raise ValueError("high_coefficients must have shape (n_species, 7)")
        if len(set(names)) != n_species:
            raise ValueError("species_names must be unique")
        if not np.all(np.isfinite(molecular_weights)) or np.any(molecular_weights <= 0.0):
            raise ValueError("molecular_weights must be finite and positive")
        if not np.all(np.isfinite(temperature_ranges)):
            raise ValueError("temperature_ranges must be finite")
        if np.any(temperature_ranges[:, 0] >= temperature_ranges[:, 1]) or np.any(
            temperature_ranges[:, 1] >= temperature_ranges[:, 2]
        ):
            raise ValueError("each temperature range must satisfy T_low < T_mid < T_high")
        if not np.all(np.isfinite(low_coefficients)) or not np.all(np.isfinite(high_coefficients)):
            raise ValueError("NASA7 coefficients must be finite")
        if not np.isfinite(self.gas_constant) or self.gas_constant <= 0.0:
            raise ValueError("gas_constant must be finite and positive")

        for array in (molecular_weights, temperature_ranges, low_coefficients, high_coefficients):
            array.setflags(write=False)
        object.__setattr__(self, "species_names", names)
        object.__setattr__(self, "molecular_weights", molecular_weights)
        object.__setattr__(self, "temperature_ranges", temperature_ranges)
        object.__setattr__(self, "low_coefficients", low_coefficients)
        object.__setattr__(self, "high_coefficients", high_coefficients)
        object.__setattr__(self, "gas_constant", float(self.gas_constant))

    @property
    def n_species(self) -> int:
        return len(self.species_names)

    @property
    def common_temperature_bounds(self) -> tuple[float, float]:
        lower = float(np.max(self.temperature_ranges[:, 0]))
        upper = float(np.min(self.temperature_ranges[:, 2]))
        if lower >= upper:
            raise ValueError("species NASA7 validity ranges have no common interval")
        return lower, upper

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Nasa7ThermoData":
        if "temperature_ranges" in value:
            ranges = value["temperature_ranges"]
        else:
            ranges = np.column_stack(
                [
                    value["temperature_low"],
                    value["temperature_mid"],
                    value["temperature_high"],
                ]
            )
        return cls(
            species_names=tuple(value["species_names"]),
            molecular_weights=np.asarray(value["molecular_weights"], dtype=np.float64),
            temperature_ranges=np.asarray(ranges, dtype=np.float64),
            low_coefficients=np.asarray(value["low_coefficients"], dtype=np.float64),
            high_coefficients=np.asarray(value["high_coefficients"], dtype=np.float64),
            gas_constant=float(value.get("gas_constant", 8314.46261815324)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "model": "NASA7",
            "species_names": list(self.species_names),
            "molecular_weights": self.molecular_weights.tolist(),
            "temperature_ranges": self.temperature_ranges.tolist(),
            "low_coefficients": self.low_coefficients.tolist(),
            "high_coefficients": self.high_coefficients.tolist(),
            "gas_constant": self.gas_constant,
            "units": {
                "molecular_weights": "kg/kmol",
                "temperature": "K",
                "gas_constant": "J/(kmol K)",
                "specific_enthalpy": "J/kg",
            },
        }


@dataclass(frozen=True)
class SpeciesOnlyClosureContract:
    """Artifact metadata for host-owned enthalpy/temperature closure.

    A conforming deployment predicts species only. The host evaluates mixture
    enthalpy before replacing species and recovers temperature after replacing
    species. NASA7 calculations in this module are an offline comparison and
    are not asserted to be bitwise or thermodynamically identical to a host
    solver such as Fluent.
    """

    schema_version: int = 1
    predicts_temperature: bool = False
    target_enthalpy_source: str = "host_pre_species_replacement"
    temperature_recovery: str = "host_post_species_replacement"
    offline_thermo_model: str = "ideal-gas-nasa7"
    exact_host_equivalence: bool = False

    def __post_init__(self) -> None:
        if self.predicts_temperature:
            raise ValueError("species-only closure must not export a temperature prediction")
        if self.exact_host_equivalence:
            raise ValueError("offline NASA7 closure must not claim exact host equivalence")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "predicts_temperature": False,
            "temperature_output": "none",
            "temperature_delta_scale": None,
            "thermochemical_closure": {
                "runtime_owner": "host",
                "target_enthalpy_source": self.target_enthalpy_source,
                "temperature_recovery": self.temperature_recovery,
                "offline_thermo_model": self.offline_thermo_model,
                "exact_host_equivalence": False,
            },
        }


@dataclass(frozen=True)
class TemperatureSolveResult:
    temperature: np.ndarray | torch.Tensor
    enthalpy_residual: np.ndarray | torch.Tensor
    converged: np.ndarray | torch.Tensor
    bracketed: np.ndarray | torch.Tensor
    iterations: int


@dataclass(frozen=True)
class ThermochemicalEndpoint:
    mass_fractions: np.ndarray | torch.Tensor
    temperature: np.ndarray | torch.Tensor
    pressure: np.ndarray | torch.Tensor
    enthalpy: np.ndarray | torch.Tensor
    density: np.ndarray | torch.Tensor
    mixture_molecular_weight: np.ndarray | torch.Tensor
    temperature_solve: TemperatureSolveResult


@dataclass(frozen=True)
class OfflineThermoComparison:
    """NASA7-vs-host comparison without an exact-equivalence claim."""

    offline_endpoint: ThermochemicalEndpoint
    host_temperature: np.ndarray | torch.Tensor
    temperature_difference: np.ndarray | torch.Tensor
    exact_host_equivalence: bool = False


def _validate_numpy_mass_fractions(
    mass_fractions: np.ndarray,
    n_species: int,
    *,
    negative_tolerance: float,
    mass_sum_tolerance: float,
) -> None:
    if mass_fractions.ndim < 1 or mass_fractions.shape[-1] != n_species:
        raise NonPhysicalStateError(f"mass fractions must end with {n_species} species")
    if not np.all(np.isfinite(mass_fractions)):
        raise NonPhysicalStateError("mass fractions must be finite")
    if np.any(mass_fractions < -negative_tolerance):
        raise NonPhysicalStateError("mass fractions contain negative values")
    if np.any(np.abs(np.sum(mass_fractions, axis=-1) - 1.0) > mass_sum_tolerance):
        raise NonPhysicalStateError("mass fractions do not sum to one")


class NumpyThermochemicalClosure:
    """Float64 NASA7 thermochemical closure for NumPy deployment paths."""

    def __init__(
        self,
        thermo_data: Nasa7ThermoData | Mapping[str, Any],
        *,
        max_iterations: int = 48,
        temperature_atol: float = 1e-7,
        temperature_rtol: float = 1e-12,
    ) -> None:
        self.data = (
            thermo_data
            if isinstance(thermo_data, Nasa7ThermoData)
            else Nasa7ThermoData.from_mapping(thermo_data)
        )
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.max_iterations = int(max_iterations)
        self.temperature_atol = float(temperature_atol)
        self.temperature_rtol = float(temperature_rtol)

    def _coefficients(self, temperature: np.ndarray) -> np.ndarray:
        use_low = temperature[..., None] <= self.data.temperature_ranges[:, 1]
        return np.where(
            use_low[..., None],
            self.data.low_coefficients,
            self.data.high_coefficients,
        )

    def species_enthalpy(self, temperature: Any) -> np.ndarray:
        """Return species mass-specific enthalpies in J/kg."""

        temperature = np.asarray(temperature, dtype=np.float64)
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise NonPhysicalStateError("temperature must be finite and positive")
        coefficients = self._coefficients(temperature)
        t = temperature[..., None]
        h_over_rt = (
            coefficients[..., 0]
            + coefficients[..., 1] * t / 2.0
            + coefficients[..., 2] * t**2 / 3.0
            + coefficients[..., 3] * t**3 / 4.0
            + coefficients[..., 4] * t**4 / 5.0
            + coefficients[..., 5] / t
        )
        return self.data.gas_constant * t * h_over_rt / self.data.molecular_weights

    def species_cp(self, temperature: Any) -> np.ndarray:
        """Return species mass-specific heat capacities in J/(kg K)."""

        temperature = np.asarray(temperature, dtype=np.float64)
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise NonPhysicalStateError("temperature must be finite and positive")
        coefficients = self._coefficients(temperature)
        t = temperature[..., None]
        cp_over_r = (
            coefficients[..., 0]
            + coefficients[..., 1] * t
            + coefficients[..., 2] * t**2
            + coefficients[..., 3] * t**3
            + coefficients[..., 4] * t**4
        )
        return self.data.gas_constant * cp_over_r / self.data.molecular_weights

    def mixture_enthalpy(self, temperature: Any, mass_fractions: Any) -> np.ndarray:
        y = np.asarray(mass_fractions, dtype=np.float64)
        if y.ndim < 1 or y.shape[-1] != self.data.n_species:
            raise ValueError(f"mass fractions must end with {self.data.n_species} species")
        return np.sum(y * self.species_enthalpy(temperature), axis=-1)

    def mixture_cp(self, temperature: Any, mass_fractions: Any) -> np.ndarray:
        y = np.asarray(mass_fractions, dtype=np.float64)
        if y.ndim < 1 or y.shape[-1] != self.data.n_species:
            raise ValueError(f"mass fractions must end with {self.data.n_species} species")
        return np.sum(y * self.species_cp(temperature), axis=-1)

    def mixture_molecular_weight(self, mass_fractions: Any) -> np.ndarray:
        y = np.asarray(mass_fractions, dtype=np.float64)
        if y.ndim < 1 or y.shape[-1] != self.data.n_species:
            raise ValueError(f"mass fractions must end with {self.data.n_species} species")
        mass_sum = np.sum(y, axis=-1)
        molar_sum = np.sum(y / self.data.molecular_weights, axis=-1)
        if np.any(~np.isfinite(molar_sum)) or np.any(molar_sum <= 0.0):
            raise NonPhysicalStateError("composition has nonpositive molar density")
        return mass_sum / molar_sum

    def density(self, temperature: Any, pressure: Any, mass_fractions: Any) -> np.ndarray:
        temperature = np.asarray(temperature, dtype=np.float64)
        pressure = np.asarray(pressure, dtype=np.float64)
        if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0.0):
            raise NonPhysicalStateError("pressure must be finite and positive")
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise NonPhysicalStateError("temperature must be finite and positive")
        return pressure * self.mixture_molecular_weight(mass_fractions) / (
            self.data.gas_constant * temperature
        )

    def reaction_endpoint(
        self,
        mass_fractions: Any,
        reaction_extent: Any,
        net_stoichiometric_matrix: Any,
    ) -> np.ndarray:
        """Apply ``Y_next = Y_now + (W S) xi`` entirely in float64."""

        y = np.asarray(mass_fractions, dtype=np.float64)
        xi = np.asarray(reaction_extent, dtype=np.float64)
        stoich = np.asarray(net_stoichiometric_matrix, dtype=np.float64)
        if stoich.ndim != 2 or stoich.shape[0] != self.data.n_species:
            raise ValueError("net_stoichiometric_matrix must have shape (n_species, n_reactions)")
        if xi.ndim < 1 or xi.shape[-1] != stoich.shape[1]:
            raise ValueError("reaction_extent must end with n_reactions")
        if y.shape[:-1] != xi.shape[:-1] or y.shape[-1] != self.data.n_species:
            raise ValueError("mass_fractions and reaction_extent batch shapes must match")
        mass_stoich = self.data.molecular_weights[:, None] * stoich
        return y + np.einsum("...r,sr->...s", xi, mass_stoich)

    def _temperature_bounds(
        self,
        batch_shape: tuple[int, ...],
        temperature_bounds: tuple[Any, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if temperature_bounds is None:
            lower_value, upper_value = self.data.common_temperature_bounds
        else:
            lower_value, upper_value = temperature_bounds
        lower = np.broadcast_to(np.asarray(lower_value, dtype=np.float64), batch_shape).copy()
        upper = np.broadcast_to(np.asarray(upper_value, dtype=np.float64), batch_shape).copy()
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower <= 0.0):
            raise TemperatureBracketError("temperature bounds must be finite and positive")
        if np.any(lower >= upper):
            raise TemperatureBracketError("lower temperature bound must be below upper bound")
        return lower, upper

    def solve_temperature(
        self,
        target_enthalpy: Any,
        mass_fractions: Any,
        *,
        temperature_bounds: tuple[Any, Any] | None = None,
        out_of_bracket: OutOfBracketMode = "raise",
        require_convergence: bool = True,
    ) -> TemperatureSolveResult:
        """Solve ``h(T, Y) = target_enthalpy`` with safeguarded Newton steps."""

        if out_of_bracket not in {"raise", "clamp", "nan"}:
            raise ValueError("out_of_bracket must be 'raise', 'clamp', or 'nan'")
        y = np.asarray(mass_fractions, dtype=np.float64)
        if y.ndim < 1 or y.shape[-1] != self.data.n_species:
            raise ValueError(f"mass fractions must end with {self.data.n_species} species")
        batch_shape = y.shape[:-1]
        target = np.broadcast_to(np.asarray(target_enthalpy, dtype=np.float64), batch_shape).copy()
        if np.any(~np.isfinite(target)):
            raise NonPhysicalStateError("target enthalpy must be finite")
        lower, upper = self._temperature_bounds(batch_shape, temperature_bounds)
        h_lower = self.mixture_enthalpy(lower, y)
        h_upper = self.mixture_enthalpy(upper, y)
        if np.any(h_upper <= h_lower):
            raise TemperatureBracketError("mixture enthalpy is not increasing over the bracket")

        bracketed = (target >= h_lower) & (target <= h_upper)
        if out_of_bracket == "raise" and np.any(~bracketed):
            raise TemperatureBracketError("target enthalpy is outside the temperature bracket")
        solve_target = np.clip(target, h_lower, h_upper)
        fraction = (solve_target - h_lower) / (h_upper - h_lower)
        temperature = lower + fraction * (upper - lower)
        converged_solve = np.zeros(batch_shape, dtype=bool)
        completed = 0

        for completed in range(1, self.max_iterations + 1):
            enthalpy = self.mixture_enthalpy(temperature, y)
            residual = enthalpy - solve_target
            tolerance = self.temperature_atol + self.temperature_rtol * np.abs(solve_target)
            converged_solve |= np.abs(residual) <= tolerance
            if np.all(converged_solve):
                break
            active = ~converged_solve
            upper = np.where(active & (residual > 0.0), temperature, upper)
            lower = np.where(active & (residual <= 0.0), temperature, lower)
            heat_capacity = self.mixture_cp(temperature, y)
            newton = temperature - residual / heat_capacity
            midpoint = 0.5 * (lower + upper)
            valid_newton = (
                np.isfinite(newton)
                & np.isfinite(heat_capacity)
                & (heat_capacity > 0.0)
                & (newton > lower)
                & (newton < upper)
            )
            candidate = np.where(valid_newton, newton, midpoint)
            temperature = np.where(active, candidate, temperature)

        solve_residual = self.mixture_enthalpy(temperature, y) - solve_target
        solve_tolerance = self.temperature_atol + self.temperature_rtol * np.abs(solve_target)
        converged_solve = np.abs(solve_residual) <= solve_tolerance
        failed_inside_bracket = bracketed & ~converged_solve
        if require_convergence and np.any(failed_inside_bracket):
            raise TemperatureConvergenceError("temperature solve failed inside a valid bracket")
        if out_of_bracket == "nan":
            temperature = np.where(bracketed, temperature, np.nan)

        requested_residual = self.mixture_enthalpy(temperature, y) - target
        requested_tolerance = self.temperature_atol + self.temperature_rtol * np.abs(target)
        converged = bracketed & (np.abs(requested_residual) <= requested_tolerance)
        return TemperatureSolveResult(
            temperature=np.asarray(temperature, dtype=np.float64),
            enthalpy_residual=np.asarray(requested_residual, dtype=np.float64),
            converged=np.asarray(converged, dtype=bool),
            bracketed=np.asarray(bracketed, dtype=bool),
            iterations=completed,
        )

    def close_endpoint(
        self,
        mass_fractions: Any,
        target_enthalpy: Any,
        pressure: Any,
        *,
        temperature_bounds: tuple[Any, Any] | None = None,
        out_of_bracket: OutOfBracketMode = "raise",
        require_convergence: bool = True,
        negative_tolerance: float = 0.0,
        mass_sum_tolerance: float = 1e-10,
    ) -> ThermochemicalEndpoint:
        y = np.asarray(mass_fractions, dtype=np.float64)
        _validate_numpy_mass_fractions(
            y,
            self.data.n_species,
            negative_tolerance=negative_tolerance,
            mass_sum_tolerance=mass_sum_tolerance,
        )
        pressure_array = np.broadcast_to(np.asarray(pressure, dtype=np.float64), y.shape[:-1]).copy()
        if np.any(~np.isfinite(pressure_array)) or np.any(pressure_array <= 0.0):
            raise NonPhysicalStateError("pressure must be finite and positive")
        solve = self.solve_temperature(
            target_enthalpy,
            y,
            temperature_bounds=temperature_bounds,
            out_of_bracket=out_of_bracket,
            require_convergence=require_convergence,
        )
        enthalpy = self.mixture_enthalpy(solve.temperature, y)
        mixture_weight = self.mixture_molecular_weight(y)
        density = pressure_array * mixture_weight / (self.data.gas_constant * solve.temperature)
        return ThermochemicalEndpoint(
            mass_fractions=y,
            temperature=solve.temperature,
            pressure=pressure_array,
            enthalpy=enthalpy,
            density=density,
            mixture_molecular_weight=mixture_weight,
            temperature_solve=solve,
        )

    def close_reaction_endpoint(
        self,
        mass_fractions: Any,
        reaction_extent: Any,
        net_stoichiometric_matrix: Any,
        target_enthalpy: Any,
        pressure: Any,
        **closure_options: Any,
    ) -> ThermochemicalEndpoint:
        next_y = self.reaction_endpoint(mass_fractions, reaction_extent, net_stoichiometric_matrix)
        return self.close_endpoint(next_y, target_enthalpy, pressure, **closure_options)

    def compare_offline_to_host_temperature(
        self,
        mass_fractions: Any,
        host_target_enthalpy: Any,
        pressure: Any,
        host_temperature: Any,
        **closure_options: Any,
    ) -> OfflineThermoComparison:
        """Compare NASA7 recovery with a host-recovered temperature.

        ``host_target_enthalpy`` should be evaluated by the host before species
        replacement. ``host_temperature`` should be recovered by the host after
        replacement. The returned difference diagnoses thermo-model mismatch;
        it is not an exact Fluent replay assertion.
        """

        endpoint = self.close_endpoint(
            mass_fractions,
            host_target_enthalpy,
            pressure,
            **closure_options,
        )
        host_temperature_array = np.broadcast_to(
            np.asarray(host_temperature, dtype=np.float64), endpoint.temperature.shape
        ).copy()
        return OfflineThermoComparison(
            offline_endpoint=endpoint,
            host_temperature=host_temperature_array,
            temperature_difference=endpoint.temperature - host_temperature_array,
        )

    def enthalpy_target_from_increment(
        self,
        temperature: Any,
        mass_fractions: Any,
        specific_enthalpy_increment: Any,
    ) -> np.ndarray:
        return self.mixture_enthalpy(temperature, mass_fractions) + np.asarray(
            specific_enthalpy_increment, dtype=np.float64
        )

    def endpoint_enthalpy_increment(
        self,
        current_temperature: Any,
        current_mass_fractions: Any,
        next_temperature: Any,
        next_mass_fractions: Any,
    ) -> np.ndarray:
        return self.mixture_enthalpy(next_temperature, next_mass_fractions) - self.mixture_enthalpy(
            current_temperature, current_mass_fractions
        )

    def specific_chemical_heat_release_rate(
        self,
        reference_temperature: Any,
        current_mass_fractions: Any,
        next_mass_fractions: Any,
        time_interval: Any,
    ) -> np.ndarray:
        """Return ``-sum(h_i delta_Y_i) / dt`` in W/kg."""

        current_y = np.asarray(current_mass_fractions, dtype=np.float64)
        next_y = np.asarray(next_mass_fractions, dtype=np.float64)
        dt = np.asarray(time_interval, dtype=np.float64)
        if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
            raise ValueError("time_interval must be finite and positive")
        chemical_increment = np.sum(
            self.species_enthalpy(reference_temperature) * (next_y - current_y), axis=-1
        )
        return -chemical_increment / dt

    def volumetric_chemical_heat_release_rate(
        self,
        reference_temperature: Any,
        pressure: Any,
        current_mass_fractions: Any,
        next_mass_fractions: Any,
        time_interval: Any,
    ) -> np.ndarray:
        specific_rate = self.specific_chemical_heat_release_rate(
            reference_temperature,
            current_mass_fractions,
            next_mass_fractions,
            time_interval,
        )
        return self.density(reference_temperature, pressure, current_mass_fractions) * specific_rate


class TorchThermochemicalClosure(nn.Module):
    """Torch float64 NASA7 closure with registered artifact constants."""

    def __init__(
        self,
        thermo_data: Nasa7ThermoData | Mapping[str, Any],
        *,
        max_iterations: int = 32,
        temperature_atol: float = 1e-7,
        temperature_rtol: float = 1e-12,
    ) -> None:
        super().__init__()
        data = (
            thermo_data
            if isinstance(thermo_data, Nasa7ThermoData)
            else Nasa7ThermoData.from_mapping(thermo_data)
        )
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        lower, upper = data.common_temperature_bounds
        self.species_names = data.species_names
        self.n_species = data.n_species
        self.max_iterations = int(max_iterations)
        self.temperature_atol = float(temperature_atol)
        self.temperature_rtol = float(temperature_rtol)
        self.gas_constant = float(data.gas_constant)
        self.register_buffer("molecular_weights", torch.tensor(data.molecular_weights, dtype=torch.float64))
        self.register_buffer(
            "default_temperature_bounds", torch.tensor([lower, upper], dtype=torch.float64)
        )
        self.register_buffer(
            "temperature_mid", torch.tensor(data.temperature_ranges[:, 1], dtype=torch.float64)
        )
        self.register_buffer("low_coefficients", torch.tensor(data.low_coefficients, dtype=torch.float64))
        self.register_buffer("high_coefficients", torch.tensor(data.high_coefficients, dtype=torch.float64))

    def _as_float64(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.molecular_weights.device, dtype=torch.float64)
        return torch.as_tensor(value, device=self.molecular_weights.device, dtype=torch.float64)

    def _coefficients(self, temperature: torch.Tensor) -> torch.Tensor:
        use_low = temperature[..., None] <= self.temperature_mid
        return torch.where(use_low[..., None], self.low_coefficients, self.high_coefficients)

    def species_enthalpy(self, temperature: Any) -> torch.Tensor:
        temperature = self._as_float64(temperature)
        coefficients = self._coefficients(temperature)
        t = temperature[..., None]
        h_over_rt = (
            coefficients[..., 0]
            + coefficients[..., 1] * t / 2.0
            + coefficients[..., 2] * t.square() / 3.0
            + coefficients[..., 3] * t.pow(3) / 4.0
            + coefficients[..., 4] * t.pow(4) / 5.0
            + coefficients[..., 5] / t
        )
        return self.gas_constant * t * h_over_rt / self.molecular_weights

    def species_cp(self, temperature: Any) -> torch.Tensor:
        temperature = self._as_float64(temperature)
        coefficients = self._coefficients(temperature)
        t = temperature[..., None]
        cp_over_r = (
            coefficients[..., 0]
            + coefficients[..., 1] * t
            + coefficients[..., 2] * t.square()
            + coefficients[..., 3] * t.pow(3)
            + coefficients[..., 4] * t.pow(4)
        )
        return self.gas_constant * cp_over_r / self.molecular_weights

    def mixture_enthalpy(self, temperature: Any, mass_fractions: Any) -> torch.Tensor:
        y = self._as_float64(mass_fractions)
        return torch.sum(y * self.species_enthalpy(temperature), dim=-1)

    def mixture_cp(self, temperature: Any, mass_fractions: Any) -> torch.Tensor:
        y = self._as_float64(mass_fractions)
        return torch.sum(y * self.species_cp(temperature), dim=-1)

    def mixture_molecular_weight(self, mass_fractions: Any) -> torch.Tensor:
        y = self._as_float64(mass_fractions)
        mass_sum = torch.sum(y, dim=-1)
        molar_sum = torch.sum(y / self.molecular_weights, dim=-1)
        return mass_sum / molar_sum

    def density(self, temperature: Any, pressure: Any, mass_fractions: Any) -> torch.Tensor:
        temperature_tensor = self._as_float64(temperature)
        pressure_tensor = self._as_float64(pressure)
        return pressure_tensor * self.mixture_molecular_weight(mass_fractions) / (
            self.gas_constant * temperature_tensor
        )

    def reaction_endpoint(
        self,
        mass_fractions: Any,
        reaction_extent: Any,
        net_stoichiometric_matrix: Any,
    ) -> torch.Tensor:
        y = self._as_float64(mass_fractions)
        xi = self._as_float64(reaction_extent)
        stoich = self._as_float64(net_stoichiometric_matrix)
        mass_stoich = self.molecular_weights[:, None] * stoich
        return y + xi @ mass_stoich.T

    def _validate_mass_fractions(
        self,
        mass_fractions: torch.Tensor,
        *,
        negative_tolerance: float,
        mass_sum_tolerance: float,
    ) -> None:
        if mass_fractions.ndim < 1 or mass_fractions.shape[-1] != self.n_species:
            raise NonPhysicalStateError(f"mass fractions must end with {self.n_species} species")
        if bool(torch.any(~torch.isfinite(mass_fractions)).detach().cpu()):
            raise NonPhysicalStateError("mass fractions must be finite")
        if bool(torch.any(mass_fractions < -negative_tolerance).detach().cpu()):
            raise NonPhysicalStateError("mass fractions contain negative values")
        mass_error = torch.abs(torch.sum(mass_fractions, dim=-1) - 1.0)
        if bool(torch.any(mass_error > mass_sum_tolerance).detach().cpu()):
            raise NonPhysicalStateError("mass fractions do not sum to one")

    def _temperature_bounds(
        self,
        target: torch.Tensor,
        temperature_bounds: tuple[Any, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temperature_bounds is None:
            lower_value: Any = self.default_temperature_bounds[0]
            upper_value: Any = self.default_temperature_bounds[1]
        else:
            lower_value, upper_value = temperature_bounds
        lower = torch.broadcast_to(self._as_float64(lower_value), target.shape).clone()
        upper = torch.broadcast_to(self._as_float64(upper_value), target.shape).clone()
        return lower, upper

    def solve_temperature(
        self,
        target_enthalpy: Any,
        mass_fractions: Any,
        *,
        temperature_bounds: tuple[Any, Any] | None = None,
        out_of_bracket: OutOfBracketMode = "raise",
        require_convergence: bool = True,
        validate: bool = True,
    ) -> TemperatureSolveResult:
        if out_of_bracket not in {"raise", "clamp", "nan"}:
            raise ValueError("out_of_bracket must be 'raise', 'clamp', or 'nan'")
        y = self._as_float64(mass_fractions)
        target = torch.broadcast_to(self._as_float64(target_enthalpy), y.shape[:-1])
        lower, upper = self._temperature_bounds(target, temperature_bounds)
        if validate:
            if bool(torch.any(~torch.isfinite(target)).detach().cpu()):
                raise NonPhysicalStateError("target enthalpy must be finite")
            invalid_bounds = (~torch.isfinite(lower)) | (~torch.isfinite(upper)) | (lower <= 0.0) | (
                lower >= upper
            )
            if bool(torch.any(invalid_bounds).detach().cpu()):
                raise TemperatureBracketError("temperature bounds must be finite, positive, and ordered")

        h_lower = self.mixture_enthalpy(lower, y)
        h_upper = self.mixture_enthalpy(upper, y)
        if validate and bool(torch.any(h_upper <= h_lower).detach().cpu()):
            raise TemperatureBracketError("mixture enthalpy is not increasing over the bracket")
        bracketed = (target >= h_lower) & (target <= h_upper)
        if out_of_bracket == "raise" and validate and bool(torch.any(~bracketed).detach().cpu()):
            raise TemperatureBracketError("target enthalpy is outside the temperature bracket")

        solve_target = torch.clamp(target, min=h_lower, max=h_upper)
        fraction = (solve_target - h_lower) / (h_upper - h_lower)
        temperature = lower + fraction * (upper - lower)
        converged_solve = torch.zeros_like(target, dtype=torch.bool)

        for _ in range(self.max_iterations):
            enthalpy = self.mixture_enthalpy(temperature, y)
            residual = enthalpy - solve_target
            tolerance = self.temperature_atol + self.temperature_rtol * torch.abs(solve_target)
            converged_solve = converged_solve | (torch.abs(residual) <= tolerance)
            active = ~converged_solve
            upper = torch.where(active & (residual > 0.0), temperature, upper)
            lower = torch.where(active & (residual <= 0.0), temperature, lower)
            heat_capacity = self.mixture_cp(temperature, y)
            newton = temperature - residual / heat_capacity
            midpoint = 0.5 * (lower + upper)
            valid_newton = (
                torch.isfinite(newton)
                & torch.isfinite(heat_capacity)
                & (heat_capacity > 0.0)
                & (newton > lower)
                & (newton < upper)
            )
            candidate = torch.where(valid_newton, newton, midpoint)
            temperature = torch.where(active, candidate, temperature)

        solve_residual = self.mixture_enthalpy(temperature, y) - solve_target
        solve_tolerance = self.temperature_atol + self.temperature_rtol * torch.abs(solve_target)
        converged_solve = torch.abs(solve_residual) <= solve_tolerance
        failed_inside_bracket = bracketed & ~converged_solve
        if require_convergence and validate and bool(torch.any(failed_inside_bracket).detach().cpu()):
            raise TemperatureConvergenceError("temperature solve failed inside a valid bracket")
        if out_of_bracket == "nan":
            temperature = torch.where(bracketed, temperature, torch.full_like(temperature, float("nan")))

        requested_residual = self.mixture_enthalpy(temperature, y) - target
        requested_tolerance = self.temperature_atol + self.temperature_rtol * torch.abs(target)
        converged = bracketed & (torch.abs(requested_residual) <= requested_tolerance)
        return TemperatureSolveResult(
            temperature=temperature,
            enthalpy_residual=requested_residual,
            converged=converged,
            bracketed=bracketed,
            iterations=self.max_iterations,
        )

    def close_endpoint(
        self,
        mass_fractions: Any,
        target_enthalpy: Any,
        pressure: Any,
        *,
        temperature_bounds: tuple[Any, Any] | None = None,
        out_of_bracket: OutOfBracketMode = "raise",
        require_convergence: bool = True,
        validate: bool = True,
        negative_tolerance: float = 0.0,
        mass_sum_tolerance: float = 1e-10,
    ) -> ThermochemicalEndpoint:
        y = self._as_float64(mass_fractions)
        if validate:
            self._validate_mass_fractions(
                y,
                negative_tolerance=negative_tolerance,
                mass_sum_tolerance=mass_sum_tolerance,
            )
        pressure_tensor = torch.broadcast_to(self._as_float64(pressure), y.shape[:-1])
        if validate and bool(
            torch.any((~torch.isfinite(pressure_tensor)) | (pressure_tensor <= 0.0)).detach().cpu()
        ):
            raise NonPhysicalStateError("pressure must be finite and positive")
        solve = self.solve_temperature(
            target_enthalpy,
            y,
            temperature_bounds=temperature_bounds,
            out_of_bracket=out_of_bracket,
            require_convergence=require_convergence,
            validate=validate,
        )
        enthalpy = self.mixture_enthalpy(solve.temperature, y)
        mixture_weight = self.mixture_molecular_weight(y)
        density = pressure_tensor * mixture_weight / (self.gas_constant * solve.temperature)
        return ThermochemicalEndpoint(
            mass_fractions=y,
            temperature=solve.temperature,
            pressure=pressure_tensor,
            enthalpy=enthalpy,
            density=density,
            mixture_molecular_weight=mixture_weight,
            temperature_solve=solve,
        )

    def close_reaction_endpoint(
        self,
        mass_fractions: Any,
        reaction_extent: Any,
        net_stoichiometric_matrix: Any,
        target_enthalpy: Any,
        pressure: Any,
        **closure_options: Any,
    ) -> ThermochemicalEndpoint:
        next_y = self.reaction_endpoint(mass_fractions, reaction_extent, net_stoichiometric_matrix)
        return self.close_endpoint(next_y, target_enthalpy, pressure, **closure_options)

    def compare_offline_to_host_temperature(
        self,
        mass_fractions: Any,
        host_target_enthalpy: Any,
        pressure: Any,
        host_temperature: Any,
        **closure_options: Any,
    ) -> OfflineThermoComparison:
        """Compare differentiable NASA7 recovery with host temperature output."""

        endpoint = self.close_endpoint(
            mass_fractions,
            host_target_enthalpy,
            pressure,
            **closure_options,
        )
        host_temperature_tensor = torch.broadcast_to(
            self._as_float64(host_temperature), endpoint.temperature.shape
        )
        return OfflineThermoComparison(
            offline_endpoint=endpoint,
            host_temperature=host_temperature_tensor,
            temperature_difference=endpoint.temperature - host_temperature_tensor,
        )

    def enthalpy_target_from_increment(
        self,
        temperature: Any,
        mass_fractions: Any,
        specific_enthalpy_increment: Any,
    ) -> torch.Tensor:
        return self.mixture_enthalpy(temperature, mass_fractions) + self._as_float64(
            specific_enthalpy_increment
        )

    def endpoint_enthalpy_increment(
        self,
        current_temperature: Any,
        current_mass_fractions: Any,
        next_temperature: Any,
        next_mass_fractions: Any,
    ) -> torch.Tensor:
        return self.mixture_enthalpy(next_temperature, next_mass_fractions) - self.mixture_enthalpy(
            current_temperature, current_mass_fractions
        )

    def specific_chemical_heat_release_rate(
        self,
        reference_temperature: Any,
        current_mass_fractions: Any,
        next_mass_fractions: Any,
        time_interval: Any,
    ) -> torch.Tensor:
        current_y = self._as_float64(current_mass_fractions)
        next_y = self._as_float64(next_mass_fractions)
        dt = self._as_float64(time_interval)
        chemical_increment = torch.sum(
            self.species_enthalpy(reference_temperature) * (next_y - current_y), dim=-1
        )
        return -chemical_increment / dt

    def volumetric_chemical_heat_release_rate(
        self,
        reference_temperature: Any,
        pressure: Any,
        current_mass_fractions: Any,
        next_mass_fractions: Any,
        time_interval: Any,
    ) -> torch.Tensor:
        return self.density(reference_temperature, pressure, current_mass_fractions) * (
            self.specific_chemical_heat_release_rate(
                reference_temperature,
                current_mass_fractions,
                next_mass_fractions,
                time_interval,
            )
        )

    def forward(
        self,
        mass_fractions: torch.Tensor,
        target_enthalpy: torch.Tensor,
        pressure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Artifact-friendly closure returning only tensor values."""

        endpoint = self.close_endpoint(
            mass_fractions,
            target_enthalpy,
            pressure,
            out_of_bracket="clamp",
            require_convergence=False,
            validate=False,
        )
        return (
            endpoint.temperature,
            endpoint.enthalpy,
            endpoint.density,
            endpoint.mixture_molecular_weight,
            endpoint.temperature_solve.bracketed,
        )
