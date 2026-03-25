# IPA-specific initialization (not generic enough for Onion core)

function ipa_point_weights_init!(weights)
    weights .= eltype(weights)(0.541324854612918)
    return weights
end
