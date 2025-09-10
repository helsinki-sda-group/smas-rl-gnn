def combine(components: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) * components.get(k, 0.0) for k in components)
