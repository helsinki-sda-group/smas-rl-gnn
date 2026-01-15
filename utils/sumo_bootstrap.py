# src/utils/sumo_bootstrap.py
from __future__ import annotations
import os, sys
from typing import Any, Callable, List, Optional, Tuple

def _imports() -> Tuple[Any, Callable[[str], str]]:
    if "SUMO_HOME" not in os.environ:
        raise RuntimeError("Set SUMO_HOME to your SUMO installation, e.g. export SUMO_HOME=/opt/sumo")
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    import traci  # type: ignore
    from sumolib import checkBinary  # type: ignore
    return traci, checkBinary

def _build_args(sumocfg_path: str, extra_args: Optional[List[str]]) -> List[str]:
    args = ["-c", sumocfg_path, "--start"]
    if extra_args:
        args += extra_args
    return args

def start_sumo(sumocfg_path: str, use_gui: bool = False, extra_args: Optional[List[str]] = None) -> Any:
    traci, checkBinary = _imports()
    binary = checkBinary("sumo-gui" if use_gui else "sumo")
    if traci.isLoaded():
        traci.close(wait=True)
    traci.start([binary, *_build_args(sumocfg_path, extra_args)])
    return traci

def make_reset_fn(sumocfg_path: str, use_gui: bool = False, extra_args: Optional[List[str]] = None) -> Callable[[], None]:
    """Reset using traci.load when possible (faster), else restart."""
    def _reset() -> None:
        traci, checkBinary = _imports()
        args = _build_args(sumocfg_path, extra_args)
        if traci.isLoaded():
            # reload in the same process/port
            traci.load(args)
        else:
            binary = checkBinary("sumo-gui" if use_gui else "sumo")
            traci.start([binary, *args])
    return _reset

def start_sumo_conn(sumocfg_path: str, use_gui: bool = False, extra_args: Optional[List[str]] = None) -> Any:
    traci, checkBinary = _imports()
    binary = checkBinary("sumo-gui" if use_gui else "sumo")
    if traci.isLoaded():
        traci.close(wait=True)
    traci.start([binary, *_build_args(sumocfg_path, extra_args)])
    return traci.getConnection()

def make_reset_fn_conn(
    set_conn: Callable[[Any], None],
    sumocfg_path: str,
    use_gui: bool = False,
    extra_args: Optional[List[str]] = None,
) -> Callable[[], None]:
    """Reset and hand a fresh connection back via set_conn(conn)."""
    def _reset() -> None:
        traci, checkBinary = _imports()
        args = _build_args(sumocfg_path, extra_args)
        if traci.isLoaded():
            try:
                traci.load(args)
            except Exception:
                # If reload fails, do full restart
                try:
                    traci.close(wait=True)
                except Exception:
                    pass
                traci.start([binary, *args])
        else:
            binary = checkBinary("sumo-gui" if use_gui else "sumo")
            traci.start([binary, *args])
        set_conn(traci.getConnection())
    return _reset
