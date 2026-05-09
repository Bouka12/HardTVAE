from abc import ABC
from collections import OrderedDict
import pandas as pd

class Measures(ABC):
    """
    Base class for measures (aka meta-features). Each measure should be implemented as a separate method.
    """

    _measures_dict: dict

    @property
    def logger(self):
        raise NotImplementedError

    def _call_method(self, name, **kwargs):
        return getattr(self, name)(**kwargs)

    def calculate_all(self, measures_list=None):
        if measures_list is None:
            measures_list = self._measures_dict.keys()
        elif isinstance(measures_list, list):
            measures_list = sorted(list(set(measures_list) & set(self._measures_dict.keys())))
        else:
            raise TypeError(f"Expected type list for parameter 'measures_list', not '{type(measures_list)}'")

        results = OrderedDict()
        for k in measures_list:
            self.logger.info(f"Calculating measure {repr(k)}")
            results[k] = self._call_method(self._measures_dict[k])

        df_measures = pd.DataFrame(results)
        return df_measures.add_prefix('feature_')