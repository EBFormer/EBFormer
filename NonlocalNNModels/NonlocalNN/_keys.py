from typing import Final

from nequip.data import register_fields

FREQUENCIES_KEY: Final[str] = "frequency"
FOURIER_KEY: Final[str] = "fourier_tranformed"
LDOS_KEY: Final[str] = "ldos"
MEAN_DOS_KEY: Final[str] = "mean_dos"
MEAN_DOS_AT_ENERGY_KEY: Final[str] = "mean_dos_at_energy"

LJx_KEY: Final[str] = "lJx"
MEAN_Jx_KEY: Final[str] = "mean_Jx"

ENERGY_BINS_KEY: Final[str] = 'energy_bins'
DOS_KEY = "dos"

DUMMY_NODE_FEATURES_KEY: Final[str] = "dummy_node_features"
DUMMY_EDGE_FEATURES_KEY: Final[str] = "dummy_edge_features"

CLEAVING_NORMAL_AXES_KEY: Final[str] = "cleaving_normal_axes"


register_fields(node_fields=[LDOS_KEY, LJx_KEY], graph_fields=[MEAN_DOS_KEY, DOS_KEY, MEAN_DOS_AT_ENERGY_KEY, MEAN_Jx_KEY,
                                                      ENERGY_BINS_KEY, DUMMY_NODE_FEATURES_KEY,
                                                      DUMMY_EDGE_FEATURES_KEY, CLEAVING_NORMAL_AXES_KEY])
