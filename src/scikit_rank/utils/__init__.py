"""Internal utilities for scikit_rank.

Modules:

* :mod:`scikit_rank.utils.ipc_materializer` - streaming Arrow IPC writer used
  by lazy preprocessing pipelines.
* :mod:`scikit_rank.utils.module_parser` - tiny ``"name:k=v;k=v"`` spec parser
  shared by losses, encoders and other configurable submodules.
"""

from scikit_rank.utils.ipc_materializer import IpcMaterializer
from scikit_rank.utils.module_parser import ModuleParserSpec

__all__ = ["IpcMaterializer", "ModuleParserSpec"]
