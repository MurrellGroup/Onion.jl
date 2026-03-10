using Onion
using ParallelTestRunner

const init_code = quote
    using Onion, Test, LinearAlgebra, Statistics
    using Einops: einsum, @einops_str
    using SpecialFunctions: erf
end

testsuite = find_tests(@__DIR__)

args = parse_args(ARGS)
if filter_tests!(testsuite, args)
    cutile = get(ENV, "ONION_TEST_CUTILE", "false") == "true"
    nnop = get(ENV, "ONION_TEST_NNOP", "false") == "true"
    filter!(testsuite) do (test, _)
        startswith(test, "ext_cutile/") && return cutile
        startswith(test, "ext_nnop/") && return nnop
        return true
    end
end

runtests(Onion, args; init_code, testsuite)
