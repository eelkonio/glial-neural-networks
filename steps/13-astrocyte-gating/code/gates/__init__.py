"""Astrocyte gate variants for three-factor learning rule.

Three variants of increasing complexity:
- BinaryGate (Variant A): 0/1 based on calcium threshold
- DirectionalGate (Variant B): signed signal from activity prediction error
- VolumeTeachingGate (Variant C): spatially-diffused teaching signal
"""

from code.gates.binary_gate import BinaryGate
from code.gates.directional_gate import DirectionalGate
from code.gates.volume_teaching import VolumeTeachingGate

__all__ = ["BinaryGate", "DirectionalGate", "VolumeTeachingGate"]
