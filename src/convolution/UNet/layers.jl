"""
    GaussianFourierProjection(embed_dim::Int, scale::T=32.0f0)

Creates a Gaussian Fourier feature projection for time embeddings. Used in diffusion models.

# Arguments
- `embed_dim::Int`: Embedding dimension. Should be even.
- `scale::T=32.0f0`: Scaling factor for the random weights.
"""
struct GaussianFourierProjection{T,S<:AbstractArray{T}}
    W::S
    scale::T
end

@layer GaussianFourierProjection
Flux.trainable(::GaussianFourierProjection) = (;)

function GaussianFourierProjection(embed_dim::Int, scale::T=32.0f0) where T<:AbstractFloat
    W = randn(T, embed_dim ÷ 2) .* scale
    GaussianFourierProjection(W, scale)
end

function (layer::GaussianFourierProjection{T})(t) where T
    t_proj = t' .* layer.W * T(2π)
    vcat(sin.(t_proj), cos.(t_proj))
end

"""
    TimeEmbedding(embed_dim::Int, num_classes::Int, embedding_dim::Int)

Creates time and optional class embeddings for diffusion models.

# Arguments
- `embed_dim::Int`: Output dimension for time embeddings
- `num_classes::Int`: Number of classes for conditional generation
- `embedding_dim::Int`: Dimension for class embeddings

# Examples
```julia
time_emb = Onion.UNet.TimeEmbedding(256, 10, 128)
t = randn(Float32, 16)
labels = rand(1:10, 16)
h = time_emb(t, labels)
```
"""
struct TimeEmbedding{P,D,E,L}
    proj::P
    dense::D
    embedding::E
    label_dense::L
end

function TimeEmbedding(embed_dim::Int, num_classes::Int, embedding_dim::Int)
    TimeEmbedding(
        GaussianFourierProjection(embed_dim),
        Chain(Dense(embed_dim, embed_dim, swish)),
        Flux.Embedding(num_classes, embedding_dim),
        Chain(Dense(embedding_dim, embed_dim, swish))
    )
end

@layer TimeEmbedding

function (te::TimeEmbedding)(t)
    h = te.proj(t)
    te.dense(h)
end

function (te::TimeEmbedding)(t, labels)
    h_time = te.proj(t)
    h_time = te.dense(h_time)

    h_label = te.embedding(labels)
    h_label = te.label_dense(h_label)

    h_time .+ h_label
end

"""
    ResidualBlock(channels::Int; kernel_size=3, time_emb=false, emb_dim=256, dropout=0.0, activation=relu)

A ResNet-style residual block with optional time embeddings, dropout, and configurable activation.

# Arguments
- `channels::Int`: Number of input and output channels
- `kernel_size=3`: Size of convolutional kernel
- `time_emb=false`: Whether to use time embeddings
- `emb_dim=256`: Dimension of time embeddings
- `dropout=0.0`: Dropout probability (0.0 means no dropout)
- `activation=relu`: Activation function to use (e.g., relu, swish, etc.)

# Examples
```julia
# Basic block with dropout
rb = Onion.UNet.ResidualBlock(64, dropout=0.1)

# Block with time embeddings and custom activation
rb = Onion.UNet.ResidualBlock(64, time_emb=true, emb_dim=256, dropout=0.1, activation=swish)

# Usage
h = randn(Float32, 32, 32, 64, 1)
t = randn(Float32, 256, 1)
h = rb(h, t)
```
"""
struct ResidualBlock{C1,C2,N1,N2,D,DO,A}
    conv1::C1
    conv2::C2
    norm1::N1
    norm2::N2
    time_mlp::D
    dropout::DO
    activation::A
end

@layer ResidualBlock

function ResidualBlock(channels::Int; kernel_size=3, time_emb=false, emb_dim=256, dropout=0.0, activation=relu)
    return ResidualBlock(
        Conv((kernel_size, kernel_size), channels=>channels, pad=1),
        Conv((kernel_size, kernel_size), channels=>channels, pad=1),
        BatchNorm(channels),
        BatchNorm(channels),
        time_emb ? Dense(emb_dim, channels, swish) : identity,
        Dropout(dropout),
        activation
    )
end

function (rb::ResidualBlock)(x)
    identity = x
    out = rb.conv1(x)
    out = rb.norm1(out)
    out = rb.activation.(out)
    out = rb.dropout(out)
    out = rb.conv2(out)
    out = rb.norm2(out)
    out = out + identity
    out = rb.activation.(out)
    rb.dropout(out)
end

function (rb::ResidualBlock)(x, t)
    identity = x
    out = rb.conv1(x)
    out = rb.norm1(out)
    out = rb.activation.(out)
    out = rb.dropout(out)
    t_proj = rb.time_mlp(t)
    t_emb = reshape(t_proj, (1, 1, size(t_proj)...))
    out = out .+ t_emb
    out = rb.conv2(out)
    out = rb.norm2(out)
    out = out + identity
    out = rb.activation.(out)
    rb.dropout(out)
end

"""
    EncoderBlock(in_channels::Int, out_channels::Int; time_emb=false, emb_dim=256, dropout=0.0, activation=relu)

An encoder block for UNet architecture with optional time embeddings and dropout.

# Arguments
- `in_channels::Int`: Number of input channels
- `out_channels::Int`: Number of output channels
- `time_emb=false`: Whether to use time embeddings
- `emb_dim=256`: Dimension of time embeddings
- `dropout=0.0`: Dropout probability (0.0 means no dropout)
- `activation=relu`: Activation function to use

# Examples
```julia
enc = Onion.UNet.EncoderBlock(3, 64, time_emb=true, emb_dim=256, dropout=0.1)
h = randn(Float32, 32, 32, 3, 1)
t = randn(Float32, 256, 1)
skip, h = enc(h, t)
```
"""
struct EncoderBlock{C,N,R,P,D,A}
    conv::C  # First convolution before residual block needed to adapt channels (green arrow in diagram)
    norm::N
    residual::R
    pool::P
    time_mlp::D
    activation::A
end

@layer EncoderBlock

function EncoderBlock(in_channels::Int, out_channels::Int; time_emb=false, emb_dim=256, dropout=0.0, activation=relu)
    return EncoderBlock(
        Conv((3, 3), in_channels=>out_channels, pad=1),
        BatchNorm(out_channels),
        ResidualBlock(out_channels, time_emb=time_emb, emb_dim=emb_dim, dropout=dropout, activation=activation),
        MaxPool((2,2)),
        time_emb ? Dense(emb_dim, out_channels, swish) : identity,
        activation
    )
end

function (eb::EncoderBlock)(x)
    x = eb.conv(x)
    x = eb.norm(x)
    x = eb.activation.(x)
    x = eb.residual(x)
    return x, eb.pool(x)
end

function (eb::EncoderBlock)(x, t)
    x = eb.conv(x)
    x = eb.norm(x)
    x = eb.activation.(x)
    t_proj = eb.time_mlp(t)
    t_emb = reshape(t_proj, (1, 1, size(t_proj)...))
    x = x .+ t_emb
    x = eb.residual(x, t)
    return x, eb.pool(x)
end

"""
    Bottleneck(channels::Int; time_emb=false, emb_dim=256, dropout=0.0, activation=relu)

A bottleneck block for UNet architecture with optional time embeddings and dropout.

# Arguments
- `channels::Int`: Number of input and output channels
- `time_emb=false`: Whether to use time embeddings
- `emb_dim=256`: Dimension of time embeddings
- `dropout=0.0`: Dropout probability (0.0 means no dropout)
- `activation=relu`: Activation function to use

# Examples
```julia
bn = Onion.UNet.Bottleneck(256, time_emb=true, emb_dim=256, dropout=0.2)
h = randn(Float32, 8, 8, 256, 1)
t = randn(Float32, 256, 1)
h = bn(h, t)
```
"""
struct Bottleneck{C1,N1,R,D,A}
    conv1::C1
    norm1::N1
    residual::R
    time_mlp::D
    activation::A
end

@layer Bottleneck

function Bottleneck(channels::Int; time_emb=false, emb_dim=256, dropout=0.0, activation=relu)
    return Bottleneck(
        Conv((3, 3), channels=>channels, pad=1),
        BatchNorm(channels),
        ResidualBlock(channels, time_emb=time_emb, emb_dim=emb_dim, dropout=dropout, activation=activation),
        time_emb ? Dense(emb_dim, channels, swish) : identity,
        activation
    )
end

function (b::Bottleneck)(x)
    x = b.conv1(x)
    x = b.norm1(x)
    x = b.activation.(x)
    b.residual(x)
end

function (b::Bottleneck)(x, t)
    x = b.conv1(x)
    x = b.norm1(x)
    x = b.activation.(x)
    t_proj = b.time_mlp(t)
    t_emb = reshape(t_proj, (1, 1, size(t_proj)...))
    x = x .+ t_emb
    b.residual(x, t)
end

"""
    DecoderBlock(in_channels::Int, out_channels::Int; time_emb=false, emb_dim=256, dropout=0.0, activation=relu)

A decoder block for UNet architecture with optional time embeddings and dropout.

# Arguments
- `in_channels::Int`: Number of input channels
- `out_channels::Int`: Number of output channels
- `time_emb=false`: Whether to use time embeddings
- `emb_dim=256`: Dimension of time embeddings
- `dropout=0.0`: Dropout probability (0.0 means no dropout)
- `activation=relu`: Activation function to use

# Examples
```julia
dec = Onion.UNet.DecoderBlock(256, 128, time_emb=true, emb_dim=256, dropout=0.1)
h = randn(Float32, 8, 8, 256, 1)
skip = randn(Float32, 16, 16, 128, 1)
t = randn(Float32, 256, 1)
h = dec(h, skip, t)
```
"""
struct DecoderBlock{U,C,N,R,D,A}
    upsample::U
    conv::C
    norm::N
    residual::R
    time_mlp::D
    activation::A
end

@layer DecoderBlock

function DecoderBlock(in_channels::Int, out_channels::Int; time_emb=false, emb_dim=256, dropout=0.0, activation=relu)
    return DecoderBlock(
        Upsample(:bilinear, scale=2),
        Conv((3, 3), in_channels * 2=>out_channels, pad=1),
        BatchNorm(out_channels),
        ResidualBlock(out_channels, time_emb=time_emb, emb_dim=emb_dim, dropout=dropout, activation=activation),
        time_emb ? Dense(emb_dim, out_channels, swish) : identity,
        activation
    )
end

function (db::DecoderBlock)(x, skip_connection)
    x = db.upsample(x)
    x = cat(x, skip_connection, dims=3)
    x = db.conv(x)
    x = db.norm(x)
    x = db.activation.(x)
    db.residual(x)
end

function (db::DecoderBlock)(x, skip_connection, t)
    x = db.upsample(x)
    x = cat(x, skip_connection, dims=3)
    x = db.conv(x)
    x = db.norm(x)
    x = db.activation.(x)
    t_proj = db.time_mlp(t)
    t_emb = reshape(t_proj, (1, 1, size(t_proj)...))
    x = x .+ t_emb
    db.residual(x, t)
end
