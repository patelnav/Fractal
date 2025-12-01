### Key Points
- **Potential Novelty**: Your proposed architecture appears novel, as no 2024-2025 papers exactly decouple boundary detection from assignment followed by bidirectional refinement conditioned on embeddings and diffusion-style iteration. However, overlaps exist in conditioning boundaries on embeddings (e.g., via voice activity detection or ASR integration) and multi-stage pipelines that refine segments post-initial detection.
- **Closest Overlaps**: Research suggests promise in conditioning on embeddings for precise transitions, as seen in approaches like diarization-conditioned ASR, but these are often joint rather than fully decoupled. Iterative refinement is rare for boundaries specifically, though diffusion models are applied to related audio tasks like synthesis and enhancement, hinting at untapped potential for boundary prediction.
- **Promising Directions**: Explore self-supervised or pruned models for sample efficiency, and cascade architectures from commercial systems for robust multi-stage refinement. Bidirectional attention in transformers (e.g., in end-to-end models) could inspire your Stage 3, while diffusion's iterative denoising might adapt to boundary localization with minimal labeled data.

### Overlaps with Proposed Approach
Your idea of speaker-agnostic change points (Stage 1), global clustering (Stage 2), and bidirectional refinement conditioned on embeddings (Stage 3) with potential diffusion iteration aligns partially with recent trends. For instance, conditioning on embeddings for boundary localization is emerging in voice activity detection and ASR-integrated diarization, but typically not as a separate refinement stage. Diffusion-based iteration is more common in audio generation than segmentation, but could extend to boundaries via denoising priors. No papers directly invalidate your approach, but they suggest enhancements like domain adaptation for noisy data or pruning for efficiency.

### State-of-the-Art Performance
Current SOTA focuses on end-to-end neural diarization (EEND) variants and hybrid models. On key benchmarks:
- **DIHARD III**: EEND-TA holds SOTA at 14.49% DER (with fine-tuning), improving on prior by scaling pre-training to 8 speakers. Mamba-based models achieve 16.0% DER with domain adaptation.
- **VoxConverse**: EEND-TA at 14.29% DER. Mamba at 9.9% (with 0.25s collar).
- **AMI**: EEND-TA at 11.04% (MixHeadset) and 15.16% (Single Distant Mic). Mamba at 18.5% DER.
Boundary errors (e.g., JER) are less emphasized, but loose boundaries in evaluation can inflate DER by up to 20% in multi-speaker scenarios.

### Commercial Insights
Commercial diarizers often outperform open-source in noisy, real-world settings due to proprietary training data and optimizations. ElevenLabs Scribe supports up to 32 speakers with word-level timestamps, but details are sparse beyond structured outputs. AssemblyAI's updates focus on embedding improvements for short, noisy segments (30% better accuracy). Deepgram uses a cascade (segmentation-embeddings-clustering) trained on 100k+ voices for language-agnostic robustness. No reverse-engineering reveals boundary-specific refinement, but their scalability suggests value in your decoupled stages.

---

Recent advancements in speaker diarization from 2024-2025 emphasize end-to-end integration, efficiency, and robustness to multi-speaker scenarios, but your proposed decoupling of boundary detection from assignment, followed by bidirectional embedding-conditioned refinement and potential diffusion iteration, remains underexplored. This survey synthesizes key papers and commercial developments, highlighting overlaps, gaps, and directions that could inform or complement your approach. We begin with boundary refinement techniques, then cover conditioning on embeddings, diffusion applications, benchmarks, commercial systems, hierarchical pipelines, and sample efficiency, drawing on arXiv preprints, conference proceedings (e.g., ICASSP 2025, Interspeech 2025), and industry blogs.

#### 1. Boundary Refinement as a Separate Stage
Boundary localization as a post-initial diarization refinement step is not widely treated separately in 2024-2025 literature, which favors joint inference to minimize error propagation. However, some pipelines implicitly refine boundaries through multi-stage processing or evaluation adjustments.

- In "Pushing the Limits of End-to-End Diarization" (September 2025), the EEND-TA model uses a non-autoregressive Transformer decoder for attractor-based diarization but omits iterative refinement for efficiency. Boundaries are derived from encoder outputs and attractors with fixed thresholds (0.5), relying on pre-training with simulated mixtures (up to 8 speakers, 80,000+ hours) rather than post-hoc iteration. This achieves low DER but could inspire your Stage 1 high-recall candidates via similar simulation.
- The Mamba-based segmentation model (March 2025) employs a local EEND stage for initial probabilities, followed by embedding extraction and clustering, effectively refining boundaries via overlapping window averaging. Longer windows (30-50s) improve boundary accuracy on complex datasets like DIHARD, but no explicit iterative step; it's coarse-to-fine in spirit, with domain adaptation boosting performance.
- Iterative approaches are scarce; "Interactive Real-Time Speaker Diarization Correction" (September 2025) uses LLMs for post-diarization transcript refinement but focuses on error correction, not boundaries. Diffusion-based boundary iteration is absent, though related works like "Can We Really Repurpose Multi-Speaker ASR Corpus" (July 2025) discuss "loose boundaries" in evaluation, noting they can reduce apparent errors by 20% without true refinement.

No papers directly invalidate your idea, but they suggest bidirectional attention (e.g., in Transformer decoders) could enhance Stage 3 without diffusion.

| Paper (Year) | Refinement Type | Iterative/Diffusion? | Key Datasets | Notes |
|--------------|-----------------|-----------------------|--------------|-------|
| EEND-TA (2025) | Attractor-based, no explicit iteration | No | AMI, VoxConverse, DIHARD | Scales pre-training; potential for high-recall candidates |
| Mamba-based (2025) | Window averaging post-segmentation | No | AMI, DIHARD, VoxConverse | Longer windows refine implicitly; SOTA on some |
| Interactive Correction (2025) | LLM-based transcript refinement | Iterative (human-in-loop) | Custom | Not boundary-focused; for real-time apps |

#### 2. Conditioning on Speaker Identity for Boundaries
Conditioning boundary detection on estimated embeddings is gaining traction, often to separate it from joint inference, aligning with your Stage 3.

- "TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings" (January 2024, but influential in 2025 citations) builds on TS-VAD, using initial embeddings to condition diarization, improving transition accuracy in overlapping speech. This decouples somewhat by assuming embeddings upfront.
- "Speaker Conditioning of Voice Activity Detection via Implicit Speaker Separation" (August 2025, Interspeech) proposes speaker-conditioned VAD, leveraging embeddings to locate transitions without joint inference. It uses implicit separation for multi-talker recordings, potentially adaptable to your bidirectional attention.
- "DiCoW: Diarization-Conditioned Whisper" (December 2024) integrates diarization with ASR, conditioning on target-speaker embeddings for precise boundaries in target-speaker extraction. "Adapting Diarization-Conditioned Whisper" (October 2025) extends this for multi-speaker ASR.
- SpeakerLM (August 2025) conditions on pre-extracted embeddings (ERes2NetV2) in an MLLM framework, supporting flexible registration for boundaries.

These suggest your conditioning could improve over joint models, especially with bidirectional refinement.

#### 3. Diffusion Models for Audio Segmentation
Diffusion is primarily for generation/synthesis, not segmentation, but iterative denoising could inspire boundary prediction.

- No direct applications to temporal boundaries in diarization; instead, see "LAPS-Diff: Diffusion-Based Framework for Singing Voice Synthesis" (July 2025) for prosody-guided diffusion. "Speech Synthesis From Continuous Features Using Per-Token Latent Diffusion" (October 2025) uses per-token diffusion for synthesis.
- Broader: NTT's ICASSP 2025 papers include diffusion for speech enhancement, estimating segments in multi-talker audio. "Towards Diverse and Efficient Audio Captioning via Diffusion Models" (Interspeech 2025) applies to segmentation-like tasks.
- X discussions highlight diffusion for audio editing (e.g., PlayDiffusion, June 2025), with iterative refinement preserving boundaries. This could extend to your diffusion-style position refinement.

#### 4. State-of-the-Art Benchmarks (2024-2025)
SOTA emphasizes scaled pre-training and hybrid architectures. Boundary metrics like JER are secondary to DER.

- EEND-TA (2025): SOTA on DIHARD III (14.49%), VoxConverse (14.29%), AMI (11.04% Mix). Architecture: Conformer encoder with CSV token.
- Mamba-based (2025): Competitive, e.g., DIHARD III (16.0%), VoxConverse (9.9%). Uses bidirectional Mamba for segmentation.
- SpeakerLM (2025): SOTA on Mandarin benchmarks (e.g., 6.60% cpCER on AliMeeting).

| Model | DER on AMI | DER on VoxConverse | DER on DIHARD III | Architecture Highlights |
|-------|------------|--------------------|-------------------|-------------------------|
| EEND-TA | 11.04% (Mix) | 14.29% | 14.49% | Scaled 8-speaker simulation |
| Mamba-based | 18.5% | 9.9% | 16.0% | Bidirectional Mamba + clustering |
| SpeakerLM | N/A (Mandarin focus) | N/A | N/A | MLLM with embedding conditioning |

#### 5. ElevenLabs Scribe and Commercial Systems
Commercial diarizers leverage massive proprietary data, outperforming open-source in noise and scalability, but technical papers are rare.

- ElevenLabs Scribe (October 2025): Handles 32 speakers, with word-level timestamps and non-speech events; no architecture details beyond high accuracy claims.
- AssemblyAI (July 2025): 30% diarization improvement via new embeddings; excels on short noisy segments (44% better at 250ms). No refinement stages disclosed.
- Deepgram (June 2024, updated 2025): Cascade architecture (segmentation, embeddings from 100k+ voices, clustering); language-agnostic, unlimited speakers. Reverse-engineering suggests focus on real-world data over academic benchmarks.

#### 6. Hierarchical/Multi-Stage Diarization
Coarse-to-fine decomposition is common, often with bidirectional elements in later stages.

- Mamba-based (2025): Hierarchical—local segmentation (coarse), embedding/clustering (fine); bidirectional Mamba enables context-aware refinement.
- "Multi-Stage Speaker Diarization for Noisy Classrooms" (2025): Explicit multi-stage for education data, refining with front-end techniques.
- SpeakerLM (2025): Multi-stage training (ASR alignment, then SDR fine-tuning) for bidirectional refinement. "M3SD" (June 2025) extends to multi-modal scenarios with cascade modules.

These validate your multi-stage idea, suggesting bidirectional attention for Stage 3.

#### 7. Sample Efficiency in Diarization
Architectural tweaks reduce labeled data needs, focusing on self-supervision and pruning.

- "Efficient and Generalizable Speaker Diarization via Structured Pruning" (November 2025): Uses distillation-guided pruning on WavLM; trains on 5-50% data while maintaining DER (e.g., 80% sparsity, 4x faster).
- "Enhanced Self-Supervised Speaker Diarization" (2025): Leverages pseudo-labels and embeddings for low-data scenarios.
- Mamba-based (2025): Trained on 1944 hours (mix of simulated/real), efficient via longer windows without augmentation.

This supports your approach if paired with simulation for Stages 1-2.

In summary, 2024-2025 research leans toward integrated, efficient models, but your decoupled, refinement-focused idea could advance the field, especially with diffusion for iteration. Consider adapting embedding conditioning from TS-SEP/DiCoW and pruning for scalability.

### Key Citations
- https://arxiv.org/abs/2509.14737
- https://hal.science/hal-05006565
- https://arxiv.org/abs/2508.06372
- https://arxiv.org/abs/2506.18623
- https://assemblyai.com/blog/speaker-diarization-update
- https://deepgram.com/learn/nextgen-speaker-diarization-and-language-detection-models
- https://elevenlabs.io/blog/meet-scribe
- https://arxiv.org/abs/2507.09226
- https://arxiv.org/abs/2509.18377
- https://arxiv.org/abs/2506.05796