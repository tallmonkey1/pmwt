"""Centralized, reproducible random-number generation.

Reproducibility is an institutional-grade requirement (SPEC §13.10). Every stochastic
component in the engine must obtain its randomness through this module rather than calling
``numpy.random`` directly or relying on global state. This guarantees:

* **Determinism** — a run is fully reproducible from its seed and config.
* **Independence** — sub-streams for different components (price sim, MM sim, RL env, ...)
  are statistically independent yet derived deterministically from one master seed, so
  components cannot accidentally share or fight over RNG state.

We use NumPy's :class:`numpy.random.SeedSequence`/:class:`Generator` (PCG64) machinery,
which is explicitly designed for spawning independent, reproducible sub-streams.
"""

from __future__ import annotations

from numpy.random import PCG64, Generator, SeedSequence

from .errors import ValidationError

__all__ = ["RandomFactory", "default_factory"]


class RandomFactory:
    """Deterministically spawns independent NumPy generators from one master seed.

    Each *named* stream is derived from the master :class:`SeedSequence` by hashing the
    stream name into a stable integer offset and spawning a child sequence. The same
    (master seed, stream name) pair always yields the same generator sequence, while
    different names yield independent streams.

    Parameters
    ----------
    seed:
        Master seed. Must be a non-negative integer. The same seed reproduces an entire
        run bit-for-bit (given identical code and inputs).
    """

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValidationError("seed must be an int", context={"seed": seed})
        if seed < 0:
            raise ValidationError("seed must be non-negative", context={"seed": seed})
        self._seed = seed
        self._root = SeedSequence(seed)
        # Cache named generators so that repeated calls for the same name return the
        # same advancing stream rather than a fresh one starting from the same state.
        self._generators: dict[str, Generator] = {}
        # Counter that guarantees uniqueness for anonymous generators requested in
        # sequence, so repeated calls to ``generator()`` without a name are independent
        # yet still reproducible for a given factory instance.
        self._anonymous_counter = 0

    @property
    def seed(self) -> int:
        """The master seed used to construct this factory."""
        return self._seed

    @staticmethod
    def _stream_offset(name: str) -> int:
        """Map a stream name to a stable non-negative 64-bit offset.

        Uses a deterministic FNV-1a hash so the mapping is identical across processes and
        Python invocations (unlike the salted builtin ``hash``).
        """
        # FNV-1a 64-bit
        fnv_offset = 0xCBF29CE484222325
        fnv_prime = 0x100000001B3
        mask = 0xFFFFFFFFFFFFFFFF
        h = fnv_offset
        for byte in name.encode("utf-8"):
            h ^= byte
            h = (h * fnv_prime) & mask
        return h

    def generator(self, name: str | None = None) -> Generator:
        """Return an independent :class:`numpy.random.Generator`.

        Parameters
        ----------
        name:
            Logical stream name (e.g. ``"rbergomi.paths"``, ``"mm_sim.quotes"``). Passing
            the same name always yields the same generator stream for a given master seed,
            which is what makes component-level results reproducible in isolation. If
            ``None``, an anonymous, sequentially-independent generator is returned.
        """
        if name is None:
            child = self._root.spawn(self._anonymous_counter + 1)[-1]
            self._anonymous_counter += 1
            return Generator(PCG64(child))
        if not name:
            raise ValidationError("stream name must be a non-empty string", context={})

        if name in self._generators:
            return self._generators[name]

        offset = self._stream_offset(name)
        # Derive a dedicated child sequence for the named stream. Combining the root
        # entropy with the name-derived offset keeps named streams independent of one
        # another and of the master seed's anonymous spawns.
        child = SeedSequence(entropy=self._seed, spawn_key=(offset,))
        gen = Generator(PCG64(child))
        self._generators[name] = gen
        return gen

    def spawn(self, name: str, count: int) -> list[Generator]:
        """Return ``count`` independent generators rooted at the named stream.

        Useful for parallel Monte-Carlo workers that each need an independent generator
        while remaining collectively reproducible.
        """
        if count <= 0:
            raise ValidationError("count must be positive", context={"count": count})
        offset = self._stream_offset(name)
        parent = SeedSequence(entropy=self._seed, spawn_key=(offset,))
        return [Generator(PCG64(child)) for child in parent.spawn(count)]


def default_factory(seed: int = 0) -> RandomFactory:
    """Convenience constructor for a :class:`RandomFactory` with an explicit seed."""
    return RandomFactory(seed)
