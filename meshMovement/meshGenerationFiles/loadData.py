import numpy as np
from pathlib import Path

def loadOrGenerateHelpers(curveCachePath: Path, generateHelperCurves: callable(int), NSample: int = 10001, curve_cache_memory: dict | None = None) -> dict:


    # 1. Already loaded this process — return immediately
    if curve_cache_memory is not None:
        return curve_cache_memory

    # 2. Disk cache exists — load it (memory-map, essentially free)
    if curveCachePath.exists():
        data = np.load(curveCachePath, allow_pickle=False)
        curves = {}
        # Keys were stored as  <side>_x  and  <side>_y
        sides = set(k.rsplit("_", 1)[0] for k in data.files)
        for side in sides:
            curves[side] = (data[f"{side}_x"], data[f"{side}_y"])
        curve_cache_memory = curves
        print(f"[RefineMesh] Loaded helper curves from cache: {curveCachePath}")
        return curves

    # 3. First run — generate and persist
    print(f"[RefineMesh] Generating helper curves (NSample={NSample}) — "
          f"will be cached to {curveCachePath}")
    curves = generateHelperCurves(NSample)
    flat = {}
    for side, (xc, yc) in curves.items():
        flat[f"{side}_x"] = np.asarray(xc)
        flat[f"{side}_y"] = np.asarray(yc)
    curveCachePath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(curveCachePath, **flat)
    print(f"[RefineMesh] Helper curves cached.")

    curve_cache_memory = curves
    return curves


