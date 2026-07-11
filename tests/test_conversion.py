from scripts.convert_hf_to_mlx import mlx_dimensions, remap_hf_key


def test_remaps_encoder_attention() -> None:
    assert (
        remap_hf_key("model.encoder.layers.3.self_attn.q_proj.weight")
        == "encoder.blocks.3.attn.query.weight"
    )


def test_remaps_decoder_cross_attention() -> None:
    assert (
        remap_hf_key("model.decoder.layers.2.encoder_attn.k_proj.weight")
        == "decoder.blocks.2.cross_attn.key.weight"
    )


def test_remaps_shared_embedding_and_skips_projection() -> None:
    assert remap_hf_key("model.decoder.embed_tokens.weight") == "decoder.token_embedding.weight"
    assert remap_hf_key("proj_out.weight") is None
    assert remap_hf_key("model.encoder.embed_positions.weight") is None


def test_dimensions() -> None:
    config = {
        "num_mel_bins": 128,
        "max_source_positions": 1500,
        "d_model": 1280,
        "encoder_attention_heads": 20,
        "encoder_layers": 32,
        "vocab_size": 51866,
        "max_target_positions": 448,
        "decoder_attention_heads": 20,
        "decoder_layers": 32,
    }
    dimensions = mlx_dimensions(config)
    assert dimensions["n_mels"] == 128
    assert dimensions["n_vocab"] == 51866
    assert dimensions["n_audio_layer"] == 32
    assert dimensions["n_text_layer"] == 32
