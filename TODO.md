# Mnemo Project TODO

## 🚀 Immediate Tasks (This Week)

### 1. Deploy Video Worker on DigitalOcean  
- [x] Create memory-optimized docker-compose for 2GB RAM server
- [x] Optimize for Podman deployment (better resource efficiency)
- [ ] Deploy using new podman-compose setup: `./scripts/deploy-server-podman.sh`
- [ ] Process the queued video (video_1751340188716362897)
- [ ] Monitor resource usage and adjust limits

### 2. Set Up Model Training Pipeline
- [ ] Choose initial model architecture for testing
- [ ] Set up training environment (free options first)
- [ ] Create dataset from processed videos
- [ ] Implement basic training loop

## 🧠 Model Training Plan

### Phase 1: Proof of Concept (Free Resources)
**Goal**: Validate the bio-inspired approach works

#### Option A: Google Colab (Recommended)
- Free GPU (T4) for 12 hours/day
- Perfect for initial experiments
- Steps:
  1. Export processed frames/audio from SQLite
  2. Upload to Google Drive
  3. Train small vision model (MobileNet/EfficientNet)
  4. Train audio model (Wav2Vec2 small)
  5. Test hierarchical gapper network

#### Option B: Kaggle Notebooks
- Free P100 GPU for 30 hours/week
- Good for longer training runs
- Can use for competitions later

#### Option C: Local Training (CPU only)
- Use your Windows machine overnight
- Train lightweight models only
- Good for prototyping architecture

### Phase 2: Specialized Models (After Validation)

#### Visual Gappers (Frame Understanding)
- **Base Model**: EfficientNet-B0 or MobileNetV3
- **Training Data**: Your extracted frames
- **Tasks**:
  - Scene detection
  - Object tracking
  - Motion analysis
  - Importance scoring

#### Audio Gappers (Sound Understanding)
- **Base Model**: Wav2Vec2-base or Whisper-tiny
- **Training Data**: Your audio segments
- **Tasks**:
  - Speech detection
  - Sound event classification
  - Audio-visual synchronization

#### Hierarchical Memory Network
- **Architecture**: Custom LSTM/Transformer hybrid
- **Purpose**: Combine gapper outputs into memories
- **Training**: Self-supervised on video sequences

### Phase 3: Production Models ($15-30/month budget)

#### DigitalOcean Spaces ($5/month)
- Store training datasets
- Save model checkpoints
- Serve trained models

#### Training Options:
1. **Vast.ai** (~$0.10-0.30/hour for RTX 3060)
   - Train for 50-150 hours/month
   - Good for iterative development

2. **RunPod** (~$0.20/hour for similar GPUs)
   - Spot instances even cheaper
   - Good for batch training

3. **Your DigitalOcean Droplet** (CPU only)
   - Fine-tune small models
   - Run inference
   - Continuous learning from new videos

## 📋 Development Roadmap

### Week 1-2: Infrastructure
- [x] Deploy orchestrator on DigitalOcean
- [ ] Deploy video worker
- [ ] Process 5-10 test videos
- [ ] Export training dataset

### Week 3-4: First Models
- [ ] Train visual importance scorer
- [ ] Train audio event detector
- [ ] Implement basic gapper network
- [ ] Test on new videos

### Month 2: Refinement
- [ ] Improve models based on results
- [ ] Add more gapper types
- [ ] Implement memory reconstruction
- [ ] Create demo interface

### Month 3: Applications
- [ ] Video search by memory
- [ ] Highlight generation
- [ ] Video summarization
- [ ] Memory-based recommendations

## 💰 Budget-Conscious Strategy

### Current Costs
- DigitalOcean Droplet: $24/month
- Total: $24/month

### Recommended Additions
1. **Month 1**: Just the droplet ($24)
2. **Month 2**: Add Spaces for storage ($29 total)
3. **Month 3**: Add GPU training budget ($50-60 total)

### Cost-Saving Tips
1. Use free resources first (Colab, Kaggle)
2. Train on subsets before full datasets
3. Use pretrained models when possible
4. Schedule GPU training in batches
5. Optimize models for CPU inference

## 🔧 Technical Implementation

### Training Data Preparation
```python
# Extract frames and audio from SQLite
# Create paired visual-audio datasets
# Generate importance labels from blur scores
# Create temporal sequences for memory training
```

### Model Architecture
```python
# Visual Gapper: CNN -> Feature Extractor
# Audio Gapper: Wav2Vec2 -> Event Classifier  
# Fusion Layer: Attention mechanism
# Memory Former: LSTM with skip connections
```

### Training Loop
```python
# Load batch of synchronized frames/audio
# Forward through gappers
# Combine features
# Predict importance/events
# Backpropagate
```

## 📊 Success Metrics

### Phase 1 Goals
- [ ] Process 10 videos successfully
- [ ] Train model with >70% accuracy on importance
- [ ] Generate meaningful video summaries
- [ ] Stay under $30/month budget

### Phase 2 Goals
- [ ] Process 100+ videos
- [ ] Multiple specialized gappers
- [ ] Real-time processing capability
- [ ] API for external applications

## 🛠️ Next Immediate Steps

1. **Fix Docker memory limits** for 2GB server
2. **Deploy video worker** on DigitalOcean
3. **Process queued video** to test full pipeline
4. **Export first dataset** for training
5. **Set up Colab notebook** for free GPU training

## 📝 Notes

- Start simple, iterate based on results
- Use free resources before paid ones
- Focus on one modality at a time
- Document everything for reproducibility
- Share progress to get feedback

Remember: The goal is to build a working bio-inspired video memory system, not perfect models. Start with basic functionality and improve iteratively!