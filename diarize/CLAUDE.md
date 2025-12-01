# Speaker Diarization Boundary Refinement: 2024-2026 Research Landscape

Your proposed architecture—decoupling boundary detection from speaker assignment with bidirectional refinement conditioned on speaker embeddings—occupies a **partially explored but underexploited research space**. Several components have precedent, but the specific combination with diffusion-style iterative position refinement appears genuinely novel. Below is a comprehensive analysis of each research question.

---

## Boundary refinement exists, but explicit position regression remains rare

The concept of treating boundary localization as a **separate refineable stage** after initial diarization has emerged in recent work, though approaches differ from your proposed architecture.

**DiaCorrect** (Han et al., ICASSP 2024) represents the most direct precedent—an error correction back-end for speaker diarization inspired by ASR spelling correction. It uses dual parallel convolutional encoders with a transformer decoder, exploiting interactions between input audio and initial diarization outputs. On simulated meeting data, DiaCorrect achieves a remarkable **62.4% DER reduction** (12.31% → 4.63%). The architecture explicitly conditions refinement on both audio features and initial system outputs, conceptually aligning with bidirectional refinement.

**"Once more Diarization"** (Boeddeker et al., Interspeech 2024) validates post-hoc segment-level speaker reassignment after initial diarization. By revisiting speaker attribution for each segment following speech enhancement, the system achieves **40%+ reduction in speaker confusion word errors**. This directly supports decoupling initial diarization from refinement.

A critical gap exists, however: most systems predict **frame-level speaker activities** rather than explicit boundary positions. Your proposed boundary regression approach—treating boundaries as continuous positions to be refined—appears underexplored. The closest work is in temporal action localization (video domain), where BRTAL (IEEE TCSVT 2025) uses offset-driven diffusion models for boundary coordinate refinement.

---

## Speaker-conditioned boundary detection has strong precedent

Using estimated speaker embeddings to guide boundary localization is **well-established** in 2024-2025 literature, validating your Stage 3 design.

**TS-SEP** (Boeddeker et al., IEEE/ACM TASLP, February 2024) explicitly conditions diarization on estimated speaker embeddings, extending TS-VAD with time-frequency resolution speaker activity estimates. Speaker embeddings guide both boundary detection and source separation, achieving state-of-the-art WER on LibriCSS. This is architecturally closest to your proposal—it decouples embedding estimation from boundary detection, then conditions boundaries on known embeddings.

**PET-TSVAD** (Microsoft, ICASSP 2024) develops profile-error-tolerant target-speaker voice activity detection, handling variable speakers with robustness to imperfect speaker profiles from initial clustering. Training with multiple clustering algorithms reduces train-test mismatch—relevant for your architecture where Stage 2 clustering produces embeddings that condition Stage 3.

**PTSD** (Jiang et al., ICASSP 2024) uses diverse prompts including speaker embeddings to localize target speech events, demonstrating the conditioning mechanism extends beyond simple TS-VAD to overlap detection and specialized diarization tasks.

The bidirectional attention conditioning you propose—using speaker embeddings on **both sides** of the boundary—appears less directly explored. Most existing work conditions on a single target speaker rather than jointly modeling the transition between two speakers.

---

## Diffusion for audio boundaries is an open research opportunity

This is where your proposed architecture may be **most novel**. Diffusion models for temporal boundary prediction in audio remain largely unexplored, despite successful applications in related domains.

**DiffSED** (Bhosale et al., arXiv August 2023) is the closest existing work—reformulating sound event detection as a generative problem where sound temporal boundaries (onset/offset) are generated from noisy proposals through denoising diffusion. A transformer decoder learns to reverse the noising process, converting noisy latent queries to ground-truth boundaries. This is conceptually identical to predicting speaker turn boundaries, but no one has yet adapted it to diarization.

**DiffGEBD** (Hwang et al., arXiv 2025) applies diffusion to generic event boundary detection in video, using temporal self-similarity encoding and classifier-free guidance to generate diverse yet accurate boundary predictions. The approach explicitly handles **boundary uncertainty**—recognizing that human annotators often disagree on exact transition points—through diverse sampling during denoising.

In video temporal action localization, **DenoiseLoc** (ICLR 2024) uses denoising diffusion specifically for refining temporal boundaries of activities. The paradigm—start with noisy boundary proposals and iteratively denoise to precise positions—maps directly to your diffusion-style refinement concept.

**Gap identified**: No paper applies denoising diffusion to speaker diarization boundaries specifically. This represents a clear opportunity. DiffSED proves the paradigm works for audio temporal boundaries; DiffGEBD shows diffusion naturally handles boundary subjectivity; your architecture could combine these insights with speaker embedding conditioning.

---

## Current SOTA: DiariZen leads open-source, commercial systems hold advantages

The benchmark landscape as of late 2024 shows **hybrid systems** (neural segmentation + clustering) outperforming pure end-to-end approaches.

| Benchmark | Best System | DER | Architecture |
|-----------|-------------|-----|--------------|
| VoxConverse | DiariZen | **5.2%** | WavLM + Conformer EEND-VC + AHC |
| AMI | DiariZen-Large + cVBx | ~18.7% | WavLM-based hybrid |
| DIHARD III | Mamba-based | ~23.5% | WavLM + Bidirectional Mamba |
| AliMeeting | Sortformer v2 | **7.0%** | Fast-Conformer + Transformer |
| Multi-dataset average | PyannoteAI (commercial) | **11.2%** | Undisclosed |

**DiariZen** (Han et al., ICASSP 2025) achieves state-of-the-art on VoxConverse by incorporating WavLM self-supervised representations into neural speaker activity detection. The system demonstrates that SSL features provide **dramatic robustness against data scarcity**—maintaining competitive performance with only 5-25% of training data.

**Mamba-based segmentation** (NTT/IRIT, arXiv October 2024) claims SOTA on three datasets (RAMC, AISHELL-4, MSDWild) by using state-space models instead of attention. Mamba's efficiency enables **30-second context windows** versus 10 seconds for LSTM, improving embedding quality for clustering.

**Sortformer v2** (NVIDIA, 2025) achieves exceptional results on Mandarin datasets (7.0% DER on AliMeeting) with **214x real-time** inference speed, but degrades with more than 4 speakers due to speaker confusion. The streaming version uses Arrival-Order Speaker Cache for online processing.

Critical observation for your architecture: **Multi-stage hybrid systems consistently outperform pure end-to-end approaches** on diverse, challenging datasets. Your proposed three-stage pipeline aligns with this trend.

---

## Commercial systems dominate through data scale, not architectural breakthroughs

Reverse-engineering commercial diarization systems reveals their advantage stems primarily from **training data scale and diversity** rather than architectural innovation.

**AssemblyAI** uses a four-stage pipeline: VAD → sliding window LSTM for d-vectors → segment-wise embedding aggregation → dual clustering (K-means + spectral with elbow method for speaker count). Recent improvements achieved **30% DER reduction** on noisy audio and **43% improvement** on very short segments (250ms) through training data expansion, not architectural changes.

**Deepgram** trains on **100,000+ unique voices** across 80+ languages with **250,000+ human-annotated examples**. Their advantage: domain diversity (meetings, podcasts, phone calls, legal, medical) and language-agnostic design. The cascade architecture (segmentation → embeddings → clustering) is conventional.

**Rev.ai Reverb** models are architecturally identical to pyannote 3.0 but fine-tuned on **26,000 hours** of diarization-specific data with precise speaker switch timings—versus public datasets' often imprecise boundaries. Rev claims the largest corpus of human-transcribed audio ever used for an open-source model (200,000 hours for ASR).

**ElevenLabs Scribe** supports up to 32 speakers with a configurable diarization threshold (0.1-0.4) controlling speaker merge/split behavior. Technical details remain scarce, but "audio understanding model" language suggests multimodal or semantic conditioning.

**Key insight for your architecture**: Commercial systems don't appear to use fundamentally different architectures—they use more data, better data, and domain-diverse training. Boundary precision improvements come from **annotation quality** (professional transcribers marking exact switch times) rather than algorithmic innovation. Your diffusion-based boundary refinement could close this gap with less data by modeling boundary uncertainty explicitly.

---

## Hierarchical/multi-stage decomposition is increasingly validated

Explicit coarse-to-fine diarization pipelines show **consistent improvements** in 2024-2025 literature.

**E-SHARC** (Singh & Ganapathy, IEEE/ACM TASLP 2025) uses hierarchical graph neural network clustering with joint optimization of embedding extractor and GNN. The system starts with segment embeddings and iteratively merges via learned GNN operations, achieving **12.6% relative improvement** on AMI, VoxConverse, and DISPLACE. VBx segmentation for boundary refinement post-clustering validates the separate-refinement paradigm.

**NSD-MS2S** (Yang et al., ICASSP 2024—CHiME-7 winner) introduces memory-aware multi-speaker embedding with Seq2Seq architecture. The Deep Interactive Module dynamically refines speaker embeddings to reduce domain mismatch, achieving **49% relative improvement** over the CHiME-7 baseline. The memory mechanism enables bidirectional information flow between stages.

**Spatio-spectral diarization** (Interspeech 2025) combines TDOA-based spatial segmentation with embedding-based clustering—demonstrating successful decoupling of boundary detection (spatial domain) from speaker assignment (spectral embeddings). This validates your Stage 1/Stage 2 decoupling concept.

**LLM-based refinement** has emerged as a post-processing layer. "From Who Said What to Who They Are" (arXiv September 2025) uses training-free LLM pipelines to reverify segments using speaker embeddings, achieving **29.7% relative error reduction** through majority voting between original, reverified, and LLM labels.

---

## Self-supervised representations enable dramatic sample efficiency

Architectural improvements—particularly **WavLM integration**—enable strong performance with limited labeled data, potentially reducing your training data requirements.

**DiariZen's critical finding**: When using WavLM representations, simulated data provides **no additional benefit**—the model achieves strong results with limited real data alone. With only **5-25% of training data**, WavLM-based systems maintain competitive performance. This fundamentally changes the training paradigm: SSL features are more valuable than more data.

**CSDA** (Coria et al., IEEE SLT 2022) enables continual self-supervised domain adaptation by processing one conversation at a time using pseudo-labels, achieving **17% relative improvement** on DIHARD III across 11 domains. The approach enables autonomous adaptation without storing sensitive data.

**JEDIS-LLM** (2024) introduces zero-shot streamable diarization through a Speaker Prompt Cache mechanism. Trained only on short audio, it generalizes to long recordings without task-specific training—suggesting that appropriate inductive biases can replace extensive labeled data.

**Child-adult diarization** (Apple/Simons Foundation, 2024) demonstrates that synthetic conversations from AudioSet enable strong zero-shot performance, with only **30 minutes of real data** fine-tuning for substantial improvements. LoRA further improves transfer efficiency.

**Architectural efficiency**: DiaPer (Landini et al., IEEE/ACM TASLP 2024) replaces LSTM-based encoder-decoder attractors with Perceiver architecture for more efficient attention over long sequences—achieving better speaker count estimation with nearly **half the inference time** on long recordings.

---

## Synthesis: your architecture's position in the research landscape

Your proposed architecture occupies a **novel but well-motivated position**:

| Component | Prior Art Status | Novelty Assessment |
|-----------|-----------------|-------------------|
| Stage 1: Speaker-agnostic boundary detection | Established (VAD, change-point detection) | Low novelty, sound foundation |
| Stage 2: Global speaker clustering | Well-established (spectral, GNN, VBx) | Low novelty, proven effective |
| Stage 3: Bidirectional attention refinement | Partial precedent (DiaCorrect, TS-SEP) | **Medium novelty**—bidirectional conditioning on *both* speakers is less explored |
| Diffusion-style iterative refinement | **No direct precedent in diarization** | **High novelty**—DiffSED proves concept for audio, but not applied to speaker boundaries |

**What might invalidate your approach**: DiaCorrect already achieves 62% DER reduction through post-hoc refinement without diffusion. If simpler refinement mechanisms suffice, the diffusion overhead may not be justified.

**What supports your approach**: Boundary precision is the primary cause of missed speech errors, not missing short segments entirely. Diffusion's ability to model uncertainty (DiffGEBD) could help with inherently ambiguous speaker transitions. Commercial systems' advantage comes partly from better boundary annotations—your approach could compensate algorithmically.

**Unexplored directions to consider**:

- **Classifier-free guidance** for boundary refinement (from DiffGEBD): Control diversity/precision tradeoff during denoising
- **Memory-aware refinement** (from NSD-MS2S): Dynamic embedding updates during boundary refinement
- **WavLM integration** for sample efficiency: Reduces training data requirements dramatically
- **GNN-based hierarchical refinement** (from E-SHARC): Learned coarse-to-fine merging instead of fixed stages
- **Temporal self-similarity encoding** (from DiffGEBD): Explicitly model speaker change dynamics for diffusion conditioning

The most promising unexplored combination: **DiffSED-style diffusion + TS-SEP-style speaker conditioning + WavLM features**. This would adapt proven diffusion mechanisms for audio boundaries to the speaker diarization task while leveraging sample-efficient representations.