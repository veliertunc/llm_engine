use crate::{
    config::ModelConfig,
    model::{LayerNorm, Linear, MultiHeadAttention, PositionalEncoding, SimpleTransformer},
};
use std::io::{Read, Write};

/// Save the entire model to disk.
pub fn save_model<W: Write>(model: &SimpleTransformer, writer: &mut W) -> std::io::Result<()> {
    save_positional(&model.pos_encoding, writer)?;

    for attn in &model.attention_layers {
        save_linear(&attn.query_proj, writer)?;
        save_linear(&attn.key_proj, writer)?;
        save_linear(&attn.value_proj, writer)?;
        save_linear(&attn.output_proj, writer)?;
    }

    for ln in &model.attn_norms {
        save_layernorm(ln, writer)?;
    }

    for ff in &model.ff_layers {
        save_linear(ff, writer)?;
    }

    for ln in &model.ff_norms {
        save_layernorm(ln, writer)?;
    }

    Ok(())
}

/// Load the model from disk.
pub fn load_model<R: Read>(
    reader: &mut R,
    config: &ModelConfig,
) -> std::io::Result<SimpleTransformer> {
    use crate::model::MultiHeadAttention;

    let pos_encoding = load_positional(reader, config.max_position_embeddings, config.hidden_size)?;

    let mut attention_layers = Vec::with_capacity(config.num_layers);
    for _ in 0..config.num_layers {
        let query_proj = load_linear(reader, config.hidden_size, config.hidden_size)?;
        let key_proj = load_linear(reader, config.hidden_size, config.hidden_size)?;
        let value_proj = load_linear(reader, config.hidden_size, config.hidden_size)?;
        let output_proj = load_linear(reader, config.hidden_size, config.hidden_size)?;

        attention_layers.push(MultiHeadAttention {
            num_heads: config.num_heads,
            head_dim: config.hidden_size / config.num_heads,
            query_proj,
            key_proj,
            value_proj,
            output_proj,
        });
    }

    let mut attn_norms = Vec::with_capacity(config.num_layers);
    for _ in 0..config.num_layers {
        attn_norms.push(load_layernorm(reader, config.hidden_size)?);
    }

    let mut ff_layers = Vec::with_capacity(config.num_layers);
    for _ in 0..config.num_layers {
        ff_layers.push(load_linear(reader, config.hidden_size, config.hidden_size)?);
    }

    let mut ff_norms = Vec::with_capacity(config.num_layers);
    for _ in 0..config.num_layers {
        ff_norms.push(load_layernorm(reader, config.hidden_size)?);
    }

    Ok(SimpleTransformer {
        hidden_size: config.hidden_size,
        pos_encoding,
        attention_layers,
        attn_norms,
        ff_layers,
        ff_norms,
    })
}

/// Serialize a Linear layer to a writer.
pub fn save_linear<W: Write>(linear: &Linear, writer: &mut W) -> std::io::Result<()> {
    for w in &linear.weights {
        writer.write_all(&w.to_le_bytes())?;
    }
    for b in &linear.biases {
        writer.write_all(&b.to_le_bytes())?;
    }
    Ok(())
}

/// Load a Linear layer from a reader.
pub fn load_linear<R: Read>(
    reader: &mut R,
    input_dim: usize,
    output_dim: usize,
) -> std::io::Result<Linear> {
    let mut weights = vec![0.0; input_dim * output_dim];
    let mut biases = vec![0.0; output_dim];

    for w in &mut weights {
        *w = read_f32(reader)?;
    }
    for b in &mut biases {
        *b = read_f32(reader)?;
    }

    Ok(Linear {
        weights,
        biases,
        input_dim,
        output_dim,
    })
}

/// Serialize a LayerNorm layer.
pub fn save_layernorm<W: Write>(ln: &LayerNorm, writer: &mut W) -> std::io::Result<()> {
    for g in &ln.gamma {
        writer.write_all(&g.to_le_bytes())?;
    }
    for b in &ln.beta {
        writer.write_all(&b.to_le_bytes())?;
    }
    Ok(())
}

/// Load a LayerNorm layer.
pub fn load_layernorm<R: Read>(reader: &mut R, dim: usize) -> std::io::Result<LayerNorm> {
    let mut gamma = vec![0.0; dim];
    let mut beta = vec![0.0; dim];

    for g in &mut gamma {
        *g = read_f32(reader)?;
    }
    for b in &mut beta {
        *b = read_f32(reader)?;
    }

    Ok(LayerNorm {
        gamma,
        beta,
        epsilon: 1e-5,
    })
}

/// Serialize a PositionalEncoding.
pub fn save_positional<W: Write>(pos: &PositionalEncoding, writer: &mut W) -> std::io::Result<()> {
    for vec in &pos.embeddings {
        for v in vec {
            writer.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Load a PositionalEncoding.
pub fn load_positional<R: Read>(
    reader: &mut R,
    max_pos: usize,
    dim: usize,
) -> std::io::Result<PositionalEncoding> {
    let mut embeddings = Vec::with_capacity(max_pos);
    for _ in 0..max_pos {
        let mut row = Vec::with_capacity(dim);
        for _ in 0..dim {
            row.push(read_f32(reader)?);
        }
        embeddings.push(row);
    }
    Ok(PositionalEncoding { embeddings })
}

/// Read a single f32 from reader.
fn read_f32<R: Read>(reader: &mut R) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
