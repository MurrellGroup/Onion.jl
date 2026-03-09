const BG_TRAINING = Ref(true)

bg_istraining() = BG_TRAINING[]

function bg_set_training!(mode::Bool)
    BG_TRAINING[] = mode
    return mode
end
