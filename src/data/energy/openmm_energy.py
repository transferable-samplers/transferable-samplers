import os

import numpy as np
import torch


def _to_numpy(x, dtype=np.float64):
    """Convert a torch.Tensor to a numpy array."""
    return x.detach().cpu().numpy().astype(dtype)


class _OpenMMEnergyGrad(torch.autograd.Function):
    """Custom autograd function that uses OpenMM forces as the backward gradient."""

    @staticmethod
    def forward(ctx, positions, evaluator):
        energies, forces = evaluator._compute(positions)
        neg_forces = torch.tensor(-forces, device=positions.device, dtype=positions.dtype)
        ctx.save_for_backward(neg_forces)
        return torch.tensor(energies, device=positions.device, dtype=positions.dtype).reshape(-1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        (neg_forces,) = ctx.saved_tensors
        # grad_output is (batch, 1); neg_forces may be (batch, n_atoms, 3)
        if neg_forces.ndim == 3:
            grad_output = grad_output.unsqueeze(-1)
        grad_input = grad_output * neg_forces
        return grad_input, None


class OpenMMEnergy:
    """Wrapper around an OpenMM system that computes energies and gradients.

    Input positions are in nm, returned energies are dimensionless (units of kT).
    Forces from OpenMM are used as the backward gradient via a custom autograd function.

    Parameters
    ----------
    system : openmm.System
        The OpenMM system object containing all force objects.
    integrator : openmm.Integrator
        A thermostated OpenMM integrator (must have a `getTemperature()` method).
    platform_name : str
        An OpenMM platform name ('CPU' or 'CUDA').
    device_index : int or None
        GPU device index for CUDA platform. Ignored for CPU.
    """

    def __init__(self, system, integrator, platform_name="CPU", device_index=None):
        from openmm import Context, Platform, unit

        self._system = system

        # Conversion factor: energy in kJ/mol -> dimensionless kT units
        self._unit_reciprocal = 1.0 / (
            integrator.getTemperature() * unit.MOLAR_GAS_CONSTANT_R
        ).value_in_unit(unit.kilojoule_per_mole)

        # Create OpenMM context
        platform = Platform.getPlatformByName(platform_name)
        properties = {}
        if platform_name == "CPU":
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
            properties["Threads"] = str(os.cpu_count() // local_world_size)
            properties["Precision"] = "mixed"
        elif platform_name == "CUDA":
            properties["Precision"] = "mixed"
            if device_index is not None:
                properties["DeviceIndex"] = str(device_index)

        self._context = Context(system, integrator, platform, properties)

    @property
    def n_atoms(self):
        return self._system.getNumParticles()

    def _compute(self, positions):
        """Compute energies and forces for a batch of positions.

        Parameters
        ----------
        positions : torch.Tensor
            Positions with shape (batch, n_atoms * 3) or (batch, n_atoms, 3).

        Returns
        -------
        energies : np.ndarray
            Energies in kT units, shape (batch,).
        forces : np.ndarray
            Forces in kT/nm units, shape matching input positions.
        """
        from openmm import unit

        orig_shape = positions.shape
        pos_np = _to_numpy(positions)

        # Reshape to (batch, n_atoms, 3)
        pos_np = pos_np.reshape(-1, self.n_atoms, 3)
        batch_size = pos_np.shape[0]

        energies = np.zeros(batch_size, dtype=np.float64)
        forces = np.zeros_like(pos_np)

        for i in range(batch_size):
            self._context.setPositions(pos_np[i])
            state = self._context.getState(getEnergy=True, getForces=True)
            e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            f = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

            if not np.isfinite(e):
                raise ValueError(f"Non-finite energy encountered: {e}")

            energies[i] = e
            forces[i] = f

        # Convert to kT units
        energies = energies * self._unit_reciprocal
        forces = forces * self._unit_reciprocal

        # Reshape forces to match input shape
        forces = forces.reshape(orig_shape)

        return energies, forces

    def __call__(self, positions):
        """Compute energies for a batch of positions.

        Parameters
        ----------
        positions : torch.Tensor
            Positions with shape (..., n_atoms, 3) or (..., n_atoms * 3).

        Returns
        -------
        energies : torch.Tensor
            Energies in kT units, shape (batch,).
        """
        orig_shape = positions.shape
        # Flatten leading batch dims
        flat = positions.reshape(-1, orig_shape[-1]) if positions.ndim > 2 else positions
        if positions.ndim > 2:
            flat = positions.reshape(-1, self.n_atoms, 3)

        result = _OpenMMEnergyGrad.apply(flat, self)

        # Restore batch dims
        batch_shape = orig_shape[:-2] if orig_shape[-2:] == (self.n_atoms, 3) else orig_shape[:-1]
        return result.reshape(*batch_shape)
