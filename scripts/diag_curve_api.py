"""
Reveal the exact MetaDrive API to FIX a Curve block's parameters (radius/angle),
so the geometry fix can be written correctly. Fast: only imports classes, builds
no environment.

Run:
    python scripts/diag_curve_api.py
"""
from __future__ import annotations


def main() -> None:
    from metadrive.component.pgblock.curve import Curve
    ps = Curve.PARAMETER_SPACE
    print("PARAMETER_SPACE type :", type(ps), "| module:", type(ps).__module__)
    print("PARAMETER_SPACE repr :", ps)
    for attr in ("parameters", "spaces", "_config", "keys"):
        val = getattr(ps, attr, "—")
        if callable(val):
            try:
                val = list(val())
            except Exception as e:
                val = f"(call failed: {e})"
        print(f"  .{attr}:", val)

    # Try to read one sample and one radius sub-space.
    try:
        print("  sample():", ps.sample())
    except Exception as e:
        print("  sample() failed:", e)

    # Locate the module that defines Box / Parameter / ParameterSpace.
    for modname in ("metadrive.component.pg_space",
                    "metadrive.component.pgblock.pg_block",
                    "metadrive.utils.space"):
        try:
            mod = __import__(modname, fromlist=["*"])
            names = [n for n in dir(mod)
                     if any(k in n for k in ("Box", "Parameter", "Space", "Config"))]
            print(f"  {modname} ->", names)
        except Exception as e:
            print(f"  {modname}: import failed ({e})")

    # What are the parameter KEYS (Parameter enum)?
    try:
        from metadrive.component.pg_space import Parameter
        print("  Parameter enum:", [p for p in dir(Parameter) if not p.startswith("_")])
    except Exception as e:
        print("  Parameter enum import failed:", e)


if __name__ == "__main__":
    main()
