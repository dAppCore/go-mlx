// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"
	"math/cmplx"

	core "dappco.re/go"
)

// The Gemma 4 audio feature extractor: raw 16 kHz waveform → the log-mel
// input_features the Conformer encoder consumes (Mantis #1839). Ported from
// the HF transformers Gemma4AudioFeatureExtractor pipeline step by step:
// truncate → pad to a sample multiple → semicausal prepend (frame/2 zeros) →
// unfold frames (frame+1 window, hop stride) → periodic Hann → rfft →
// magnitude → HTK triangular mel bank → log(mel + floor) → frame-validity
// mask (a frame is real only when its window's last sample is real audio),
// with masked frames zeroed. Pure host-side DSP — no MLX arrays; the float64
// pipeline mirrors numpy's promotion and casts to float32 at the end.

// Gemma4AudioFeatureConfig mirrors the feature_extractor section of the
// model's processor_config.json. The model is the source of truth — absent
// dimensions stay zero and fail loud at extractor build.
type Gemma4AudioFeatureConfig struct {
	FeatureSize      int32     `json:"feature_size"`
	SamplingRate     int32     `json:"sampling_rate"`
	FrameLength      int32     `json:"frame_length"`
	HopLength        int32     `json:"hop_length"`
	FFTLength        int32     `json:"fft_length"`
	FFTOverdrive     bool      `json:"fft_overdrive"`
	MinFrequency     float64   `json:"min_frequency"`
	MaxFrequency     float64   `json:"max_frequency"`
	MelFloor         float64   `json:"mel_floor"`
	Preemphasis      float64   `json:"preemphasis"`
	PreemphasisHTK   bool      `json:"preemphasis_htk_flavor"`
	Dither           float64   `json:"dither"`
	InputScaleFactor float64   `json:"input_scale_factor"`
	PaddingValue     float64   `json:"padding_value"`
	PerBinMean       []float64 `json:"per_bin_mean"`
	PerBinStddev     []float64 `json:"per_bin_stddev"`
	MaxLengthSamples int32     `json:"-"`
	PadToMultiple    int32     `json:"-"`
	FeatureExtractor string    `json:"feature_extractor_type"`
}

// gemma4ProcessorConfig is the slice of processor_config.json this package
// reads (the image/video sections belong to the vision lane).
type gemma4ProcessorConfig struct {
	AudioMsPerToken  int32                     `json:"audio_ms_per_token"`
	AudioSeqLength   int32                     `json:"audio_seq_length"`
	FeatureExtractor *Gemma4AudioFeatureConfig `json:"feature_extractor"`
}

// LoadGemma4AudioFeatureConfig reads the audio feature_extractor section from
// the model directory's processor_config.json. Returns (nil, nil) when the
// model ships no processor config (text-only checkpoints).
func LoadGemma4AudioFeatureConfig(modelPath string) (*Gemma4AudioFeatureConfig, error) {
	path := core.PathJoin(modelPath, "processor_config.json")
	read := core.ReadFile(path)
	if !read.OK {
		return nil, nil
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, core.E("gemma4.audio", "processor_config.json read returned non-byte data", nil)
	}
	var processor gemma4ProcessorConfig
	if r := core.JSONUnmarshal(data, &processor); !r.OK {
		return nil, core.E("gemma4.audio", "parse processor_config.json", nil)
	}
	return processor.FeatureExtractor, nil
}

// Gemma4AudioFeatureExtractor converts waveforms to log-mel features.
type Gemma4AudioFeatureExtractor struct {
	cfg        *Gemma4AudioFeatureConfig
	window     []float32 // periodic Hann over FrameLength
	melFilters [][]float64
	// HF __call__ defaults: clips truncate at 30 s (480 000 samples @16k)
	// and waveforms right-pad to a multiple of 128 samples.
	maxSamples    int32
	padToMultiple int32
}

// NewGemma4AudioFeatureExtractor builds the extractor from the model's
// declared feature config. Fails loud on missing dimensions or a
// non-power-of-two FFT length (the rfft below is radix-2).
func NewGemma4AudioFeatureExtractor(cfg *Gemma4AudioFeatureConfig) (*Gemma4AudioFeatureExtractor, error) {
	if cfg == nil {
		return nil, core.NewError("gemma4: audio feature config is nil")
	}
	if cfg.FeatureSize <= 0 || cfg.SamplingRate <= 0 || cfg.FrameLength <= 0 || cfg.HopLength <= 0 {
		return nil, core.E("gemma4.audio", core.Sprintf(
			"feature config incomplete: size=%d rate=%d frame=%d hop=%d",
			cfg.FeatureSize, cfg.SamplingRate, cfg.FrameLength, cfg.HopLength), nil)
	}
	fft := cfg.FFTLength
	if fft <= 0 {
		fft = 1 << int32(math.Ceil(math.Log2(float64(cfg.FrameLength))))
		if cfg.FFTOverdrive {
			fft *= 2
		}
	}
	if fft&(fft-1) != 0 || fft < cfg.FrameLength {
		return nil, core.E("gemma4.audio", core.Sprintf("fft_length %d must be a power of two ≥ frame_length %d", fft, cfg.FrameLength), nil)
	}
	if cfg.MaxFrequency <= cfg.MinFrequency {
		return nil, core.E("gemma4.audio", core.Sprintf("mel band [%v, %v] is empty", cfg.MinFrequency, cfg.MaxFrequency), nil)
	}
	resolved := *cfg
	resolved.FFTLength = fft

	// Periodic Hann, float32 like the reference (frames multiply in f32
	// before numpy's rfft promotes to f64 — kept bit-faithful).
	window := make([]float32, cfg.FrameLength)
	for n := range window {
		window[n] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/float64(cfg.FrameLength)))
	}

	maxSamples := resolved.MaxLengthSamples
	if maxSamples <= 0 {
		maxSamples = 480_000
	}
	padMultiple := resolved.PadToMultiple
	if padMultiple <= 0 {
		padMultiple = 128
	}
	return &Gemma4AudioFeatureExtractor{
		cfg:           &resolved,
		window:        window,
		melFilters:    gemma4HTKMelFilterBank(int(fft)/2+1, int(cfg.FeatureSize), cfg.MinFrequency, cfg.MaxFrequency, int(cfg.SamplingRate)),
		maxSamples:    maxSamples,
		padToMultiple: padMultiple,
	}, nil
}

// Extract converts one waveform (16 kHz mono, [-1,1] float32 samples) into
// log-mel features. Returns the features as a flat [frames × FeatureSize]
// float32 slice, the per-frame validity mask, and the frame count.
func (e *Gemma4AudioFeatureExtractor) Extract(samples []float32) ([]float32, []bool, int, error) {
	if e == nil {
		return nil, nil, 0, core.NewError("gemma4: audio feature extractor is nil")
	}
	if len(samples) == 0 {
		return nil, nil, 0, core.NewError("gemma4: empty waveform")
	}
	cfg := e.cfg
	if int32(len(samples)) > e.maxSamples {
		samples = samples[:e.maxSamples]
	}

	// Right-pad to the sample multiple; padded samples are not real audio.
	realLen := len(samples)
	padded := realLen
	if rem := padded % int(e.padToMultiple); rem != 0 {
		padded += int(e.padToMultiple) - rem
	}

	// Semicausal prepend (frame/2 zeros) so the first frame centres at t=0.
	// The waveform buffer carries [prepend ⊕ samples ⊕ right-pad]; validity
	// marks only the real samples.
	prepend := int(cfg.FrameLength) / 2
	wave := make([]float64, prepend+padded)
	valid := make([]bool, prepend+padded)
	scale := cfg.InputScaleFactor
	if scale == 0 {
		scale = 1
	}
	for i, s := range samples {
		wave[prepend+i] = float64(s) * scale
		valid[prepend+i] = true
	}

	frameSize := int(cfg.FrameLength) + 1 // unfold size; preemphasis==0 drops the last sample
	hop := int(cfg.HopLength)
	numFrames := (len(wave) - frameSize) / hop
	if (len(wave) - frameSize) >= 0 {
		numFrames++
	} else {
		numFrames = 0
	}
	if numFrames <= 0 {
		return nil, nil, 0, core.E("gemma4.audio", core.Sprintf("waveform too short: %d samples < frame %d", realLen, frameSize), nil)
	}
	if cfg.Preemphasis != 0 {
		return nil, nil, 0, core.NewError("gemma4: preemphasis extraction not implemented (no shipped Gemma 4 config uses it)")
	}

	bins := int(cfg.FFTLength)/2 + 1
	features := make([]float32, numFrames*int(cfg.FeatureSize))
	mask := make([]bool, numFrames)
	frame := make([]float64, int(cfg.FFTLength))
	spectrum := make([]complex128, int(cfg.FFTLength))

	for f := 0; f < numFrames; f++ {
		start := f * hop
		// Window in float32 (reference dtype), widen for the FFT.
		for n := 0; n < int(cfg.FrameLength); n++ {
			frame[n] = float64(float32(wave[start+n]) * e.window[n])
		}
		for n := int(cfg.FrameLength); n < int(cfg.FFTLength); n++ {
			frame[n] = 0
		}
		gemma4RFFT(frame, spectrum)

		row := features[f*int(cfg.FeatureSize) : (f+1)*int(cfg.FeatureSize)]
		for m := 0; m < int(cfg.FeatureSize); m++ {
			acc := 0.0
			filter := e.melFilters[m]
			for b := 0; b < bins; b++ {
				if filter[b] != 0 {
					acc += cmplx.Abs(spectrum[b]) * filter[b]
				}
			}
			value := math.Log(acc + cfg.MelFloor)
			if len(cfg.PerBinMean) == int(cfg.FeatureSize) {
				value -= cfg.PerBinMean[m]
			}
			if len(cfg.PerBinStddev) == int(cfg.FeatureSize) {
				value /= cfg.PerBinStddev[m]
			}
			row[m] = float32(value)
		}

		// A frame is real audio only when its window's LAST sample is —
		// masked frames zero out, mirroring the reference's mask multiply.
		mask[f] = valid[start+frameSize-1]
		if !mask[f] {
			for m := range row {
				row[m] = 0
			}
		}
	}
	return features, mask, numFrames, nil
}

// gemma4HTKMelFilterBank ports HF audio_utils.mel_filter_bank with
// mel_scale="htk", norm=nil: triangular filters over linspace'd HTK-mel
// centres, evaluated at the FFT bin frequencies. Returned mel-major
// ([numMel][bins]) for the row-dot in Extract.
func gemma4HTKMelFilterBank(bins, numMel int, minFreq, maxFreq float64, samplingRate int) [][]float64 {
	hzToMel := func(hz float64) float64 { return 2595.0 * math.Log10(1.0+hz/700.0) }
	melToHz := func(mel float64) float64 { return 700.0 * (math.Pow(10, mel/2595.0) - 1.0) }

	melMin, melMax := hzToMel(minFreq), hzToMel(maxFreq)
	filterFreqs := make([]float64, numMel+2)
	for i := range filterFreqs {
		mel := melMin + (melMax-melMin)*float64(i)/float64(numMel+1)
		filterFreqs[i] = melToHz(mel)
	}
	fftFreqs := make([]float64, bins)
	// linspace(0, samplingRate//2, bins) — integer-divided ceiling per the
	// reference (matters only for odd sampling rates).
	nyquist := float64(samplingRate / 2)
	for i := range fftFreqs {
		fftFreqs[i] = nyquist * float64(i) / float64(bins-1)
	}

	filters := make([][]float64, numMel)
	for m := range filters {
		row := make([]float64, bins)
		lower, centre, upper := filterFreqs[m], filterFreqs[m+1], filterFreqs[m+2]
		for b, freq := range fftFreqs {
			down := (freq - lower) / (centre - lower)
			up := (upper - freq) / (upper - centre)
			if v := math.Min(down, up); v > 0 {
				row[b] = v
			}
		}
		filters[m] = row
	}
	return filters
}

// gemma4RFFT computes an in-place iterative radix-2 FFT of the real input
// frame into spectrum (full complex spectrum; callers read bins [0, n/2]).
func gemma4RFFT(frame []float64, spectrum []complex128) {
	n := len(frame)
	// Bit-reversal permutation.
	for i, j := 0, 0; i < n; i++ {
		if i < j {
			spectrum[i], spectrum[j] = complex(frame[j], 0), complex(frame[i], 0)
		} else if i == j {
			spectrum[i] = complex(frame[i], 0)
		}
		mask := n >> 1
		for ; j&mask != 0; mask >>= 1 {
			j &^= mask
		}
		j |= mask
	}
	// Butterflies.
	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		step := -2 * math.Pi / float64(size)
		for start := 0; start < n; start += size {
			for k := 0; k < half; k++ {
				angle := step * float64(k)
				w := cmplx.Rect(1, angle)
				even := spectrum[start+k]
				odd := spectrum[start+k+half] * w
				spectrum[start+k] = even + odd
				spectrum[start+k+half] = even - odd
			}
		}
	}
}
