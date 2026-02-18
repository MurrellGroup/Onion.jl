"""
    Backend

Abstract type for compute backends. Backends determine which
implementations are used for primitives.

Subtypes are singleton types used for dispatch:

    struct DefaultBackend <: Backend end
"""
abstract type Backend end
