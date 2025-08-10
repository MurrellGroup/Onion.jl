using Onion
using Test
using Flux
include("attention_tests.jl")

@testset "Onion.jl" begin
    # Test UNet components
    @testset "UNet Components" begin
        # Test TimeEmbedding
        @testset "TimeEmbedding" begin
            time_emb = Onion.TimeEmbedding(256, 10, 128)
            t = randn(Float32, 16)
            labels = rand(1:10, 16)

            # Test time-only embedding
            h1 = time_emb(t)
            @test size(h1) == (256, 16)

            # Test time+class embedding
            h2 = time_emb(t, labels)
            @test size(h2) == (256, 16)
        end

        # Test ResidualBlock
        @testset "ResidualBlock" begin
            # Without time embedding
            rb1 = Onion.ResidualBlock(64)
            x = randn(Float32, 32, 32, 64, 2)
            y1 = rb1(x)
            @test size(y1) == size(x)

            # With time embedding
            rb2 = Onion.ResidualBlock(64, time_emb=true, emb_dim=256)
            t = randn(Float32, 256, 2)
            y2 = rb2(x, t)
            @test size(y2) == size(x)
        end

        # Test EncoderBlock
        @testset "EncoderBlock" begin
            eb = Onion.EncoderBlock(3, 64, time_emb=true, emb_dim=256)
            x = randn(Float32, 32, 32, 3, 2)
            t = randn(Float32, 256, 2)

            skip, out = eb(x, t)
            @test size(skip) == (32, 32, 64, 2)
            @test size(out) == (16, 16, 64, 2)
        end

        # Test FlexibleUNet
        @testset "FlexibleUNet Forward Pass" begin
            model = Onion.FlexibleUNet(
                in_channels=3,
                out_channels=3,
                depth=4,
                base_channels=16,
                channel_multipliers=[1, 2, 4, 8],
                time_embedding=true,
                num_classes=10,
                embedding_dim=32,
                time_emb_dim=64
            )

            x = randn(Float32, 32, 32, 3, 2)
            t = randn(Float32, 2)
            labels = rand(1:10, 2)

            # Test unconditional forward pass
            y1 = model(x)
            @test size(y1) == (32, 32, 3, 2)

            # Test time-conditioned forward pass
            y2 = model(x, t)
            @test size(y2) == (32, 32, 3, 2)

            # Test time+class-conditioned forward pass
            y3 = model(x, t, labels)
            @test size(y3) == (32, 32, 3, 2)
        end

        # Test helper functions
        @testset "UNet Helpers" begin
            # Test reverse_tuple
            t = (1, 2, 3, 4, 5)
            rt = Onion.reverse_tuple(t)
            @test rt == (5, 4, 3, 2, 1)

            # Test process_encoders and process_decoders with a mini-model
            enc1 = Onion.EncoderBlock(3, 16)
            enc2 = Onion.EncoderBlock(16, 32)
            encoders = (enc1, enc2)

            dec1 = Onion.DecoderBlock(32, 16)
            dec2 = Onion.DecoderBlock(16, 8)
            decoders = (dec1, dec2)

            x = randn(Float32, 32, 32, 3, 2)

            # Test process_encoders
            x_out, skips = Onion.process_encoders(x, encoders)
            @test length(skips) == 2
            @test size(x_out) == (8, 8, 32, 2)

            # Test process_decoders
            rev_skips = Onion.reverse_tuple(skips)
            y = Onion.process_decoders(x_out, decoders, rev_skips)
            @test size(y) == (32, 32, 8, 2)
        end
    end

    @testset "DyT" begin
        dyt = Onion.DyT(256)
        x = randn(Float32, 256, 2)
        y = dyt(x)
        @test size(y) == size(x)
    end
end
