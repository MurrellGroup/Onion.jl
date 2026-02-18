using OnionCore: Backend

struct DefaultBackend <: Backend end

struct NNopBackend <: Backend end

struct cuTileBackend <: Backend end
