<p align="center">
    <a target="_blank" href="https://colab.research.google.com/github/bshall/soft-vc/blob/main/soft-vc-demo.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</p>

# Acoustic-Model

Training and inference scripts for the acoustic models in [A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion](https://ieeexplore.ieee.org/abstract/document/9746484). For more details see [soft-vc](https://github.com/bshall/soft-vc). Audio samples can be found [here](https://bshall.github.io/soft-vc/). Colab demo can be found [here](https://colab.research.google.com/github/bshall/soft-vc/blob/main/soft-vc-demo.ipynb).

<div align="center">
    <img width="100%" alt="Soft-VC"
      src="https://raw.githubusercontent.com/bshall/acoustic-model/main/acoustic-model.png">
</div>
<div>
  <sup>
    <strong>Fig 1:</strong> Architecture of the voice conversion system. a) The <strong>discrete</strong> content encoder clusters audio features to produce a sequence of discrete speech units. b) The <strong>soft</strong> content encoder is trained to predict the discrete units. The acoustic model transforms the discrete/soft speech units into a target spectrogram. The vocoder converts the spectrogram into an audio waveform.
  </sup>
</div>

## Example Usage

### Programmatic Usage

```python
import torch
import numpy as np

# Load checkpoint (either hubert_soft or hubert_discrete)
acoustic = torch.hub.load("bshall/acoustic-model:main", "hubert_soft").cuda()

# Load speech units
units = torch.from_numpy(np.load("path/to/units"))

# Generate mel-spectrogram
mel = acoustic.generate(units)
```

### Script-Based Usage

```
usage: generate.py [-h] {soft,discrete} in-dir out-dir

Generate spectrograms from input speech units (discrete or soft).

positional arguments:
  {soft,discrete}  available models (HuBERT-Soft or HuBERT-Discrete)
  in-dir           path to the dataset directory.
  out-dir          path to the output directory.

optional arguments:
  -h, --help       show this help message and exit
```

## Training

### Step 1: Dataset Preparation

Download and extract the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset. The training script expects the following tree structure for the dataset directory:

```
└───wavs
    ├───dev
    │   ├───LJ001-0001.wav
    │   ├───...
    │   └───LJ050-0278.wav
    └───train
        ├───LJ002-0332.wav
        ├───...
        └───LJ047-0007.wav
```

The `train` and `dev` directories should contain the training and validation splits respectively. The splits used for the paper can be found [here](https://github.com/bshall/acoustic-model/releases/tag/v0.1).

### Step 2: Extract Spectrograms

Extract mel-spectrograms using the `mel.py` script:

```
usage: mels.py [-h] in-dir out-dir

Extract mel-spectrograms for an audio dataset.

positional arguments:
  in-dir      path to the dataset directory.
  out-dir     path to the output directory.

optional arguments:
  -h, --help  show this help message and exit
```

for example:

```
python mel.py path/to/LJSpeech-1.1/wavs path/to/LJSpeech-1.1/mels
```

At this point the directory tree should look like:

```
├───mels
│   ├───...
└───wavs
    ├───...
```

### Step 3: Extract Discrete or Soft Speech Units

Use the HuBERT-Soft or HuBERT-Discrete content encoders to extract speech units. First clone the [content encoder repo](https://github.com/bshall/hubert) and then run `encode.py` (see the repo for details):

```
usage: encode.py [-h] [--extension EXTENSION] {soft,discrete} in-dir out-dir

Encode an audio dataset.

positional arguments:
  {soft,discrete}       available models (HuBERT-Soft or HuBERT-Discrete)
  in-dir                path to the dataset directory.
  out-dir               path to the output directory.

optional arguments:
  -h, --help            show this help message and exit
  --extension EXTENSION
                        extension of the audio files (defaults to .flac).
```

for example:

```
python encode.py soft path/to/LJSpeech-1.1/wavs path/to/LJSpeech-1.1/soft --extension .wav
```

At this point the directory tree should look like:

```
├───mels
│   ├───...
├───soft/discrete
│   ├───...
└───wavs
    ├───...
```

### Step 4: Train the Acoustic-Model

```
usage: train.py [-h] [--resume RESUME] [--discrete] dataset-dir checkpoint-dir

Train the acoustic model.

positional arguments:
  dataset-dir      path to the data directory.
  checkpoint-dir   path to the checkpoint directory.

optional arguments:
  -h, --help       show this help message and exit
  --resume RESUME  path to the checkpoint to resume from.
  --discrete       Use discrete units.
```

## Links

- [Soft-VC repo](https://github.com/bshall/soft-vc)
- [Soft-VC paper](https://ieeexplore.ieee.org/abstract/document/9746484)
- [HuBERT content encoders](https://github.com/bshall/hubert)
- [HiFiGAN vocoder](https://github.com/bshall/hifigan)

## Citation

If you found this work helpful please consider citing our paper:

```
@inproceedings{
    soft-vc-2022,
    author={van Niekerk, Benjamin and Carbonneau, Marc-André and Zaïdi, Julian and Baas, Matthew and Seuté, Hugo and Kamper, Herman},
    booktitle={ICASSP}, 
    title={A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion}, 
    year={2022}
}
```

1.	A summary of the purpose and function of the code: 
Training and inference scripts for the acoustic models in A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion. The purpose of this project is to eliminate noise on the spectrum. we describe the voice conversion system we use to compare discrete and soft speech units. The system consists of three components: a content encoder, an acoustic model, and a vocoder. The content encoder extracts discrete or soft speech units from input audio. Next, the acoustic model translates the speech units into a target spectrogram. Finally, the spectrogram is converted into an audio waveform by the vocoder.


2.	Describe your innovation in code improvement:
   Increase the number of speakers at the beginning
   Upgrade the software used in the vocoder


3.	Changing and improving the source code:
   

	Import torch
import torch.nn as nn
	from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
	
	URLS = {
	"hubert-discrete": "https://github.com/bshall/acoustic-model/releases/download/v0.1/hubert-discrete-d49e1c77.pt",
	"hubert-soft": "https://github.com/bshall/acoustic-model/releases/download/v0.1/hubert-soft-0321fd7e.pt",
	}
	
	
	class AcousticModel(nn.Module):
	def __init__(self, discrete: bool = False, upsample: bool = True):
	super().__init__()
	self.encoder = Encoder(discrete, upsample)
	self.decoder = Decoder()
	
	def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
	x = self.encoder(x)
	return self.decoder(x, mels)
	
	@torch.inference_mode()
	def generate(self, x: torch.Tensor) -> torch.Tensor:
	x = self.encoder(x)
	return self.decoder.generate(x)
	
	
	class Encoder(nn.Module):
	def __init__(self, discrete: bool = False, upsample: bool = True):
	super().__init__()
	self.embedding = nn.Embedding(100 + 1, 256) if discrete else None
	self.prenet = PreNet(256, 256, 256)
	self.convs = nn.Sequential(
	nn.Conv1d(256, 512, 5, 1, 2),
	nn.ReLU(),
	nn.InstanceNorm1d(512),
	nn.ConvTranspose1d(512, 512, 4, 2, 1) if upsample else nn.Identity(),
	nn.Conv1d(512, 512, 5, 1, 2),
	nn.ReLU(),
	nn.InstanceNorm1d(512),
	nn.Conv1d(512, 512, 5, 1, 2),
	nn.ReLU(),
	nn.InstanceNorm1d(512),
	)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
	if self.embedding is not None:
	x = self.embedding(x)
	x = self.prenet(x)
	x = self.convs(x.transpose(1, 2))
	return x.transpose(1, 2)
	
	
	class Decoder(nn.Module):
	def __init__(self):
	super().__init__()
	self.prenet = PreNet(128, 256, 256)
	self.lstm1 = nn.LSTM(512 + 256, 768, batch_first=True)
	self.lstm2 = nn.LSTM(768, 768, batch_first=True)
	self.lstm3 = nn.LSTM(768, 768, batch_first=True)
	self.proj = nn.Linear(768, 128, bias=False)
	
	def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
	mels = self.prenet(mels)
	x, _ = self.lstm1(torch.cat((x, mels), dim=-1))
	res = x
	x, _ = self.lstm2(x)
	x = res + x
	res = x
	x, _ = self.lstm3(x)
	x = res + x
	return self.proj(x)
	
	@torch.inference_mode()
	def generate(self, xs: torch.Tensor) -> torch.Tensor:
	m = torch.zeros(xs.size(0), 128, device=xs.device)
	h1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
	c1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
	h2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
	c2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
	h3 = torch.zeros(1, xs.size(0), 768, device=xs.device)
	c3 = torch.zeros(1, xs.size(0), 768, device=xs.device)
	
	mel = []
	for x in torch.unbind(xs, dim=1):
	m = self.prenet(m)
	x = torch.cat((x, m), dim=1).unsqueeze(1)
	x1, (h1, c1) = self.lstm1(x, (h1, c1))
	x2, (h2, c2) = self.lstm2(x1, (h2, c2))
	x = x1 + x2
	x3, (h3, c3) = self.lstm3(x, (h3, c3))
	x = x + x3
	m = self.proj(x).squeeze(1)
	mel.append(m)
	return torch.stack(mel, dim=1)
	
	
	class PreNet(nn.Module):
	def __init__(
	self,
	input_size: int,
	hidden_size: int,
	output_size: int,
	dropout: float = 0.5,
	):
	super().__init__()
	self.net = nn.Sequential(
	nn.Linear(input_size, hidden_size),
	nn.ReLU(),
	nn.Dropout(dropout),
	nn.Linear(hidden_size, output_size),
	nn.ReLU(),
	nn.Dropout(dropout),
	)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
	return self.net(x)
	
	
	def _acoustic(
	name: str,
	discrete: bool,
	upsample: bool,
	pretrained: bool = True,
	progress: bool = True,
	) -> AcousticModel:
	acoustic = AcousticModel(discrete, upsample)
	if pretrained:
	checkpoint = torch.hub.load_state_dict_from_url(URLS[name], progress=progress)
	consume_prefix_in_state_dict_if_present(checkpoint["acoustic-model"], "module.")
	acoustic.load_state_dict(checkpoint["acoustic-model"])
	acoustic.eval()
	return acoustic
	
	
	def hubert_discrete(
	pretrained: bool = True,
	progress: bool = True,
	) -> AcousticModel:
	r"""HuBERT-Discrete acoustic model from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
	Args:
	pretrained (bool): load pretrained weights into the model
	progress (bool): show progress bar when downloading model
	"""
	return _acoustic(
	"hubert-discrete",
	discrete=True,
	upsample=True,
	pretrained=pretrained,
	progress=progress,
	)
	
	
	def hubert_soft(
	pretrained: bool = True,
	progress: bool = True,
	) -> AcousticModel:
	r"""HuBERT-Soft acoustic model from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
	Args:
	pretrained (bool): load pretrained weights into the model
	progress (bool): show progress bar when downloading model
	"""
	return _acoustic(
	"hubert-soft",
	discrete=False,
	upsample=True,
	pretrained=pretrained,
	progress=progress,
	)
This part is added to the continuation of the sourse code by Fatemeh shaker
       import torch, torchaudio
    import requests
    import IPython.display as display
Reference to this project:  https://github.com/Francis-Komizu/Sovits


4.	The result of changing and improving the code in evaluating the output audio file
 For better voice quality, in Sovits2, I utilize the f0 model used in StarGANv2-VC to get the fundamental frequency feature of an input audio and feed it to the vocoder of VITS.
	import torch
	from torch import nn
	
	class JDCNet(nn.Module):
	"""
	Joint Detection and Classification Network model for singing voice melody.
	"""
	def __init__(self, num_class=722, seq_len=31, leaky_relu_slope=0.01):
	super().__init__()
	self.seq_len = seq_len # 31
	self.num_class = num_class
	
	# input = (b, 1, 31, 513), b = batch size
	self.conv_block = nn.Sequential(
	nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False), # out: (b, 64, 31, 513)
	nn.BatchNorm2d(num_features=64),
	nn.LeakyReLU(leaky_relu_slope, inplace=True),
	nn.Conv2d(64, 64, 3, padding=1, bias=False), # (b, 64, 31, 513)
	)
	
	# res blocks
	self.res_block1 = ResBlock(in_channels=64, out_channels=128) # (b, 128, 31, 128)
	self.res_block2 = ResBlock(in_channels=128, out_channels=192) # (b, 192, 31, 32)
	self.res_block3 = ResBlock(in_channels=192, out_channels=256) # (b, 256, 31, 8)
	
	# pool block
	self.pool_block = nn.Sequential(
	nn.BatchNorm2d(num_features=256),
	nn.LeakyReLU(leaky_relu_slope, inplace=True),
	nn.MaxPool2d(kernel_size=(1, 4)), # (b, 256, 31, 2)
	nn.Dropout(p=0.5),
	)
	
	# maxpool layers (for auxiliary network inputs)
	# in = (b, 128, 31, 513) from conv_block, out = (b, 128, 31, 2)
	self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 40))
	# in = (b, 128, 31, 128) from res_block1, out = (b, 128, 31, 2)
	self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 20))
	# in = (b, 128, 31, 32) from res_block2, out = (b, 128, 31, 2)
	self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 10))
	
	# in = (b, 640, 31, 2), out = (b, 256, 31, 2)
	self.detector_conv = nn.Sequential(
	nn.Conv2d(640, 256, 1, bias=False),
	nn.BatchNorm2d(256),
	nn.LeakyReLU(leaky_relu_slope, inplace=True),
	nn.Dropout(p=0.5),
	)
	
	# input: (b, 31, 512) - resized from (b, 256, 31, 2)
	self.bilstm_classifier = nn.LSTM(
	input_size=512, hidden_size=256,
	batch_first=True, bidirectional=True) # (b, 31, 512)
	
	# input: (b, 31, 512) - resized from (b, 256, 31, 2)
	self.bilstm_detector = nn.LSTM(
	input_size=512, hidden_size=256,
	batch_first=True, bidirectional=True) # (b, 31, 512)
	
	# input: (b * 31, 512)
	self.classifier = nn.Linear(in_features=512, out_features=self.num_class) # (b * 31, num_class)
	
	# input: (b * 31, 512)
	self.detector = nn.Linear(in_features=512, out_features=2) # (b * 31, 2) - binary classifier
	
	# initialize weights
	self.apply(self.init_weights)
	
	def get_feature_GAN(self, x):
	seq_len = x.shape[-2]
	x = x.float().transpose(-1, -2)
	
	convblock_out = self.conv_block(x)
	
	resblock1_out = self.res_block1(convblock_out)
	resblock2_out = self.res_block2(resblock1_out)
	resblock3_out = self.res_block3(resblock2_out)
	poolblock_out = self.pool_block[0](resblock3_out)
	poolblock_out = self.pool_block[1](poolblock_out)
	
	return poolblock_out.transpose(-1, -2)
	
	def forward(self, x):
	"""
	Returns:
	classification_prediction, detection_prediction
	sizes: (b, 31, 722), (b, 31, 2)
	"""
	###############################
	# forward pass for classifier #
	###############################
	x = x.float().transpose(-1, -2)
	
	convblock_out = self.conv_block(x)
	
	resblock1_out = self.res_block1(convblock_out)
	resblock2_out = self.res_block2(resblock1_out)
	resblock3_out = self.res_block3(resblock2_out)
	
	
	poolblock_out = self.pool_block[0](resblock3_out)
	poolblock_out = self.pool_block[1](poolblock_out)
	GAN_feature = poolblock_out.transpose(-1, -2)
	poolblock_out = self.pool_block[2](poolblock_out)
	
	# (b, 256, 31, 2) => (b, 31, 256, 2) => (b, 31, 512)
	classifier_out = poolblock_out.permute(0, 2, 1, 3).contiguous().view((-1, self.seq_len, 512))
	classifier_out, _ = self.bilstm_classifier(classifier_out) # ignore the hidden states
	
	classifier_out = classifier_out.contiguous().view((-1, 512)) # (b * 31, 512)
	classifier_out = self.classifier(classifier_out)
	classifier_out = classifier_out.view((-1, self.seq_len, self.num_class)) # (b, 31, num_class)
	
	# sizes: (b, 31, 722), (b, 31, 2)
	# classifier output consists of predicted pitch classes per frame
	# detector output consists of: (isvoice, notvoice) estimates per frame
	return torch.abs(classifier_out.squeeze()), GAN_feature, poolblock_out
	
	@staticmethod
	def init_weights(m):
	if isinstance(m, nn.Linear):
	nn.init.kaiming_uniform_(m.weight)
	if m.bias is not None:
	nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Conv2d):
	nn.init.xavier_normal_(m.weight)
	elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
	for p in m.parameters():
	if p.data is None:
	continue
	
	if len(p.shape) >= 2:
	nn.init.orthogonal_(p.data)
	else:
	nn.init.normal_(p.data)
	
	
	class ResBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
	super().__init__()
	self.downsample = in_channels != out_channels
	
	# BN / LReLU / MaxPool layer before the conv layer - see Figure 1b in the paper
	self.pre_conv = nn.Sequential(
	nn.BatchNorm2d(num_features=in_channels),
	nn.LeakyReLU(leaky_relu_slope, inplace=True),
	nn.MaxPool2d(kernel_size=(1, 2)), # apply downsampling on the y axis only
	)
	
	# conv layers
	self.conv = nn.Sequential(
	nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
	kernel_size=3, padding=1, bias=False),
	nn.BatchNorm2d(out_channels),
	nn.LeakyReLU(leaky_relu_slope, inplace=True),
	nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
	)
	
	# 1 x 1 convolution layer to match the feature dimensions
	self.conv1by1 = None
	if self.downsample:
	self.conv1by1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
	
	def forward(self, x):
	x = self.pre_conv(x)
	if self.downsample:
	x = self.conv(x) + self.conv1by1(x)
	else:
	x = self.conv(x) + x
	return x


5.	Reference to the project link
       https://github.com/Francis-Komizu/Sovits



