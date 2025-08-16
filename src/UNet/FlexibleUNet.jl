"""
    reverse_tuple(t::Tuple)

Helper function that reverses the order of elements in a tuple.
Use for maintaining type stability when reversing the order of skip connections.
"""
@generated function reverse_tuple(t::Tuple)
    N = length(t.parameters)
    args = [:(t[$i]) for i in N:-1:1]
    return :(tuple($(args...)))
end

"""
    FlexibleUNet(;
        in_channels=3,
        out_channels=3,
        depth=3,
        base_channels=64,
        channel_multipliers=[1, 2, 4],
        time_embedding=false,
        num_classes=0,
        embedding_dim=128,
        time_emb_dim=256,
        dropout=0.0,
        dropout_depth=0,
        activation=relu
    )

A flexible UNet architecture with configurable depth and channel dimensions.
Supports optional time and class embeddings for diffusion models and conditional generation.

# Arguments
- `in_channels=3`: Number of input channels
- `out_channels=3`: Number of output channels
- `depth=3`: Number of encoder/decoder blocks
- `base_channels=64`: Base channel dimension (multiplied at each level)
- `channel_multipliers=[1, 2, 4]`: Multipliers for channel dimensions at each level
- `time_embedding=false`: Whether to use time embeddings
- `num_classes=0`: Number of class labels for conditional generation
- `embedding_dim=128`: Dimension for class embeddings
- `time_emb_dim=256`: Dimension for time embeddings
- `dropout=0.0`: Dropout probability to apply to inner layers
- `dropout_depth=0`: Number of layers to apply dropout to, starting from the innermost layers (0 means no dropout). Maximum value is 1+depth (bottleneck + all encoding/decoding levels)
- `activation=relu`: Activation function to use throughout the network

# Examples
```julia
# Basic model without dropout
model = FlexibleUNet(
    in_channels=3,
    out_channels=3,
    depth=4,
    base_channels=32,
    channel_multipliers=[1, 2, 4, 8],
    time_embedding=true
)

# Model with dropout applied to the 3 innermost layers
model = FlexibleUNet(
    in_channels=3,
    out_channels=3,
    depth=4,
    base_channels=32,
    channel_multipliers=[1, 2, 4, 8],
    time_embedding=true,
    dropout=0.2,
    dropout_depth=3
)

x = randn(Float32, 32, 32, 3, 1)
t = randn(Float32, 1)
labels = [5]
y = model(x, t, labels)
```
"""
struct FlexibleUNet{E,B,D,FC,T}
    encoders::E
    bottleneck::B
    decoders::D
    final_conv::FC
    time_embed::T
end

Flux.@layer FlexibleUNet

function FlexibleUNet(;
    in_channels=3,
    out_channels=3,
    depth=3,
    base_channels=64,
    channel_multipliers=[1, 2, 4], # Multipliers for each level
    time_embedding=false,
    num_classes=0,
    embedding_dim=128,
    time_emb_dim=256,
    dropout=0.0,
    dropout_depth=0,
    activation=relu
)
    # Ensure we have enough channel multipliers for the requested depth
    if length(channel_multipliers) < depth
        # Extend with the last multiplier (create new array, don't mutate)
        channel_multipliers = vcat(channel_multipliers,
                fill(channel_multipliers[end], depth - length(channel_multipliers)))
    elseif length(channel_multipliers) > depth
        # Trim to the requested depth (create new array, don't mutate)
        channel_multipliers = channel_multipliers[1:depth]
    end

    # Calculate actual channel numbers
    channels = [base_channels * m for m in channel_multipliers]

    # Calculate maximum possible dropout depth (bottleneck + all encoders/decoders symmetrically)
    # Since we're applying dropout symmetrically, the maximum depth is bottleneck (1) + depth

    # Limit dropout_depth to the maximum number of layers (bottleneck + depth)
    dropout_depth = min(dropout_depth, 1 + depth)

    # Bottleneck is considered the innermost layer (index 1)
    dropout_bottleneck = dropout_depth >= 1 ? dropout : 0.0

    # Create encoder blocks with appropriate dropout
    encoders = []
    input_ch = in_channels
    for i in 1:depth
        # For encoders, the deepest is the one closest to the bottleneck (index = depth)
        # Innermost encoder is at depth position, so we count from innermost outward
        # Apply dropout if layer is within dropout_depth counting from the center
        encoder_dropout = dropout_depth >= (depth - i + 1) ? dropout : 0.0

        encoder = EncoderBlock(input_ch, channels[i],
                             time_emb=time_embedding,
                             emb_dim=time_emb_dim,
                             dropout=encoder_dropout,
                             activation=activation)
        push!(encoders, encoder)
        input_ch = channels[i]
    end

    # Create bottleneck
    bottleneck = Bottleneck(channels[end],
                          time_emb=time_embedding,
                          emb_dim=time_emb_dim,
                          dropout=dropout_bottleneck,
                          activation=activation)

    # Create decoder blocks with appropriate dropout
    decoders = []
    for i in depth:-1:1
        out_ch = i > 1 ? channels[i-1] : channels[1]

        # For decoders, the deepest is the one closest to the bottleneck (first one)
        # Apply dropout symmetrically with encoders
        decoder_dropout = dropout_depth >= (depth - i + 1) ? dropout : 0.0

        decoder = DecoderBlock(channels[i], out_ch,
                             time_emb=time_embedding,
                             emb_dim=time_emb_dim,
                             dropout=decoder_dropout,
                             activation=activation)
        push!(decoders, decoder)
    end

    # Create final convolution to map to output channels
    final_conv = Conv((1, 1), channels[1]=>out_channels)

    # Create time embedding
    time_embed = time_embedding ?
                TimeEmbedding(time_emb_dim, num_classes, embedding_dim) : identity

    # Convert to tuples for type stability
    encoders_tuple = Tuple(encoders)
    decoders_tuple = Tuple(decoders)

    FlexibleUNet(encoders_tuple, bottleneck, decoders_tuple, final_conv, time_embed)
end

# Process encoders and collect skip connections without mutations
function process_encoders(x, encoders::Tuple)
    # Use foldl to accumulate skip connections in a tuple
    (final_x, skips) = foldl(encoders; init=(x, ())) do acc, encoder
        (curr_x, skip_connections) = acc
        skip, new_x = encoder(curr_x)
        (new_x, (skip_connections..., skip))
    end

    return final_x, skips
end

# Process encoders with time embedding
function process_encoders(x, t, encoders::Tuple)
    # Use foldl to accumulate skip connections in a tuple
    (final_x, skips) = foldl(encoders; init=(x, ())) do acc, encoder
        (curr_x, skip_connections) = acc
        skip, new_x = encoder(curr_x, t)
        (new_x, (skip_connections..., skip))
    end

    return final_x, skips
end

# Process decoders using skip connections
function process_decoders(x, decoders::Tuple, skip_connections::Tuple)
    # Maps 1-to-1 reverse skip connections to decoders
    # Note: skip_connections should be reversed before passing to this function

    # Check for empty case
    if isempty(decoders) || isempty(skip_connections)
        return x
    end

    # Use foldl to apply decoders with skip connections
    return foldl(zip(decoders, skip_connections); init=x) do acc_x, (decoder, skip)
        decoder(acc_x, skip)
    end
end

# Process decoders with time embedding
function process_decoders(x, t, decoders::Tuple, skip_connections::Tuple)
    # Maps 1-to-1 reverse skip connections to decoders
    # Note: skip_connections should be reversed before passing to this function

    # Check for empty case
    if isempty(decoders) || isempty(skip_connections)
        return x
    end

    # Use foldl to apply decoders with skip connections
    return foldl(zip(decoders, skip_connections); init=x) do acc_x, (decoder, skip)
        decoder(acc_x, skip, t)
    end
end

# Standard forward pass without time embedding - using foldl to avoid mutations
function (model::FlexibleUNet)(x)
    # Apply encoder blocks and collect skip connections
    x, skip_connections = process_encoders(x, model.encoders)

    # Apply bottleneck
    x = model.bottleneck(x)

    # Apply decoder blocks with skip connections (reverse skip connections)
    rev_skips = reverse_tuple(skip_connections)
    x = process_decoders(x, model.decoders, rev_skips)

    # Apply final convolution
    model.final_conv(x)
end

# Forward pass with time embedding - using foldl to avoid mutations
function (model::FlexibleUNet)(x, t::T) where T <: AbstractArray
    t = model.time_embed(t)

    # Apply encoder blocks and collect skip connections
    x, skip_connections = process_encoders(x, t, model.encoders)

    # Apply bottleneck
    x = model.bottleneck(x, t)

    # Apply decoder blocks with skip connections (reverse skip connections)
    rev_skips = reverse_tuple(skip_connections)
    x = process_decoders(x, t, model.decoders, rev_skips)

    # Apply final convolution
    model.final_conv(x)
end

# Forward pass with time embedding and class labels - using foldl to avoid mutations
function (model::FlexibleUNet)(x, t::T, labels::L) where {T <: AbstractArray, L <: AbstractArray}
    t = model.time_embed(t, labels)

    # Apply encoder blocks and collect skip connections
    x, skip_connections = process_encoders(x, t, model.encoders)

    # Apply bottleneck
    x = model.bottleneck(x, t)

    # Apply decoder blocks with skip connections (reverse skip connections)
    rev_skips = reverse_tuple(skip_connections)
    x = process_decoders(x, t, model.decoders, rev_skips)

    # Apply final convolution
    model.final_conv(x)
end