---
title: ISMIR - Paper overviews
---
 
 This year's ISMIR was great as ever, this time featuring
 
 * lots of deep learning - I suspect since it became much more easy to use with recently developed libraries 
 * lots of new, and surprisingly large, datasets (suited for the new deep learning era)
 * and a fantastic boat tour through Paris!
 
 For those that want some very quick overview about many of the papers (but not all - and the selection is biased towards my own research interests admittedly).
 I created "mini-abstracts" designed to describe the core idea or contribution in each paper that should at least be understandable to someone familiar with the field, since even abstracts tend to sometimes be wordy or unneccessarily, well... abstract! I divided them according to the ISMIR conference session they belong to.
 
 Use [this page](http://ismir2018.ircam.fr/pages/events-main-program.html) in parallel to quickly retrieve the links to each paper's PDF document.
 
# Musical objects

## (A-1) A Confidence Measure For Key Labelling

Roman B. Gebhardt, Michael Stein and Athanasios Lykartsis 

Uncertainty in key classification for songs can be estimated by looking at how much the estimated key varies across the whole song (stability), and by taking the sum of the chroma vector at each timepoint and taking the average over the whole song as a measure of how much tonality is contained (keyness)

## (A-2) Improved Chord Recognition by Combining Duration and Harmonic Language Models
Filip Korzeniowski and Gerhard Widmer 

Use a model for predicting the next chord given the previous ones, combined with a duration model that predicts at which timestep the chord changes, as a language model to facilitate the learning of long-term dependencies that would be otherwise hard to learn with a time-frame based approach.

## (A-4) A Predictive Model for Music based on Learned Interval Representations 
Stefan Lattner, Maarten Grachten and Gerhard Widmer 

Use a gated recurrent autoencoder to encode the relativ change in pitch at each timestep, then model these relativ changes with an RNN to perform monphonic pitch sequence generation, to enable an RNN to generalize better to repeating melody patterns that continually rise/fall each time.

## (A-5) An End-to-end Framework for Audio-to-Score Music Transcription on Monophonic Excerpts 
Miguel A. Román, Antonio Pertusa and Jorge Calvo-Zaragoza 

Use a neural network on audio to output a symbolic sequence using a vocbulary with clefs, keys, pitches etc. required to reconstruct a full score-sheet, using a CTC loss.

## (A-6) Evaluating Automatic Polyphonic Music Transcription 
Andrew McLeod and Mark Steedman

Proposes five metrics with which to rate the quality of a music transcription system, and combines them to one metric describing overall quality, aiming to penalize each mistake only in exactly one of the five metrics (no multiple penalties). There is however no evaluation as to how this metric correlates with subjective quality ratings by humans.

## (A-7) Onsets and Frames: Dual-Objective Piano Transcription 
Curtis Hawthorne, Erich Elsen, Jialin Song, Adam Roberts, Ian Simon, Colin Raffel, Jesse Engel, Sageev Oore and Douglas Eck 

Instead of using a frame-wise cross-entropy loss on the piano roll output for transcription, also predict the onset position of notes to improve performance/reduce spurious note activations. They also predict note velocity separately to further improve the sound of the synthesized transcription.

## (A-8) Player Vs Transcriber: A Game Approach To Data Manipulation For Automatic Drum Transcription 
Carl Southall, Ryan Stables and Jason Hockman 

Add another model to the drum transcription setting (player) that can learn to use data augmentation operations on the training set to decrease the resulting transciption accuracy. Player and transcriber are trained together to make the transcriber learn from difficult examples not seen in the training data.

## (A-10) Evaluating a collection of Sound-Tracing Data of Melodic Phrases 
Tejaswinee Kelkar, Udit Roy and Alexander Refsum Jensenius 

Make people move their bodies in response to melodic phrases while motion-capturing them and then try to find out which movement features correlate with/predict the corresponding melody.

## (A-11) Main Melody Estimation with Source-Filter NMF and CRNN 
Dogac Basaran, Slim Essid and Geoffroy Peeters 

Pretrain a source-filter NMF model to provide useful features for input into a convolutional-recurrent neural network to track the main melody in music pieces. Pretraining helps since it provides a better representation of the dominant fundamental frequency/pitch salience.

## (A-13) A single-step approach to musical tempo estimation using a convolutional neural network 
Hendrik Schreiber and Meinard Mueller 

Neural network that predicts the local tempo given a 12 second long audio input, and its aggregated outputs over a whole song can be used for estimating the whole song's tempo.

## (A-14) Analysis of Common Design Choices in Deep Learning Systems for Downbeat Tracking 
Magdalena Fuentes, Brian McFee, Hélène C. Crayencour, Slim Essid and Juan Pablo Bello 

Investigation how downbeat performance changes when the SotA approaches are changed slightly, e.g. what temporal granularity the input spectrogram has, how the output is decoded from the neural network, convolutional-RNN vs only RNN.

# Generation, visual

## (B-5) Bridging audio analysis, perception and synthesis with perceptually-regularized variational timbre spaces 
Philippe Esling, Axel Chemla--Romeu-Santos and Adrien Bitton 

Beta-VAE is used on instrument samples. The latent space is additionally regularized such that the distances between samples of different instruments corresponds to the perceived timbral difference according to perceptual ratings. The resulting model's latent space can be used to classify the instrument, pitch, dynamics and family, and together with the decoder one can synthesize smoothly interpolated new sounds.

## (B-6) Conditioning Deep Generative Raw Audio Models for Structured Automatic Music 
Rachel Manzelli, Vijay Thakkar, Ali Siahkamari and Brian Kulis 

Combine symbolic and audio music models: Recurrent network is trained to model symbolic note sequences, and a Wavenet model separately is trained to produce raw audio conditioned on a piano-roll representation. Then the models are put together to synthesize music pieces.

## (B-7) Convolutional Generative Adversarial Networks with Binary Neurons for Polyphonic Music Generation 
Hao-Wen Dong and Yi-Hsuan Yang 

To adapt GANs for symbolic music generation, which is a discrete problem and not a continuous problem as usually handled by GANs, they use the straight-through estimator ("stochastic binary neurons") that have a binary output (randomly sampled) in the forward, but a real-valued probability in the backward path to compute gradients.

# Source separation

## (C-2) Music Source Separation Using Stacked Hourglass Networks 
Sungheon Park, Taehoon Kim, Kyogu Lee and Nojun Kwak 

2D U-Net neural network for source separation applied multiple times in a row using a residual connection, so that the initial estimate can be further refined each time

## (C-3) The Northwestern University Source Separation Library 
Ethan Manilow, Prem Seetharaman and Bryan Pardo 

Library for source separation: Supports using trained separation models easily, offers computation of evaluation metrics

## (C-4) Improving Bass Saliency Estimation using Transfer Learning and Label Propagation 
Jakob Abeßer, Stefan Balke and Meinard Müller 

Detecting bass notes in jazz ensemble recordings. Two techniques are investigated in the face of the small available labelled data: Label propagation - train model on annotated dataset, then predict labels for unlabelled data and retrain - and transfer learning - network is trained on isolated bass recordings first, then on the actual jazz data.

## (C-5) Improving Peak-picking Using Multiple Time-step Loss Functions 
Carl Southall, Ryan Stables and Jason Hockman 

Since many current models that predict a series of events given an audio sequence are trained with frame-wise cross-entropy followed by separate peak picking, the models activations might not be well suited for the peak picking procedure. Loss functions that act on neighbouring outputs as well are investigated to remedy this.

## (C-6) Zero-Mean Convolutions for Level-Invariant Singing Voice Detection 
Jan Schlüter and Bernhard Lehner 

Singing voice classifiers turn out to be sensitive to the overall volume of the music output, which is undesirable. While data augmentation by random amplification and mixing of voice and instrumentals helps, the paper shows you can also directly bake in this invariance by constraining the first convolutional layer so that ist weights sum up to 1, which improves robustness a bit more. However a very quiet music input with singing voice will now be classified as positive, although a listener might not be able to hear anything and say there is no singing voice.

## (C-8) Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation 
Daniel Stoller, Sebastian Ewert and Simon Dixon 

This is our own paper ;)

Change the "U-Net", previously used in biomedical image segmentation and magnitude-based source separation to capture multi-scale features and dependencies, from 2D to 1D convolution (across time), to perform separation directly on the waveform without needing artifact-inducing source reconstruction steps.

For more information, please see the corresponding Github repository [here](https://github.com/f90/Wave-U-Net)

## (C-13) Music Mood Detection Based on Audio and Lyrics with Deep Neural Net 
Rémi Delbouys, Romain Hennequin, Francesco Piccoli, Jimena Royo-Letelier and Manuel Moussallam

Predicting the mood of a music piece by combining audio and lyrics information helps performance.

# Corpora

## (D-4) DALI: a large Dataset of synchronized Audio, LyrIcs and notes, automatically created using teacher-student machine learning paradigm 
Gabriel Meseguer-Brocal, Alice Cohen-Hadria and Geoffroy Peeters 

5000 music pieces with lyrics aligned up to a syllable-level, created by matching Karaoke files with user-defined aligned lyrics with the corresponding audio tracks according to a singing voice probability vector across the time duration of the song. Iteratively, the singing voice detection system can be retrained with the newly derived dataset to improve its performance, and the cycle can be repeated.

## (D-5) OpenMIC-2018: An open data-set for multiple instrument recognition 
Eric Humphrey, Simon Durand and Brian McFee 

Large dataset of 10 sec music snippets with labels indicating which instruments are present.

## (D-6) From Labeled to Unlabeled Data – On the Data Challenge in Automatic Drum Transcription 
Chih-Wei Wu and Alexander Lerch 

Investigating feature learning and student-teacher learning paradigms for drum transcription to circumvent the lack of labelled training data. Performance does not clearly increase however, indicating the need for better feature learning/student-teacher learning approaches to enable better transfer.

## (D-9) VocalSet: A Singing Voice Dataset 
Julia Wilkins, Prem Seetharaman, Alison Wahl and Bryan Pardo 

Solo singing voice recordings by 20 different professional singers with annotated labels of singing style and pitch.

## (D-10) The NES Music Database: A multi-instrumental dataset with expressive performance attributes 
Chris Donahue, Huanru Henry Mao and Julian McAuley 

Large dataset of polyphonic music in symbolic form taken from NES games, but with extra performance-related attributes (note velocity, timbre)

## (D-14) Revisiting Singing Voice Detection: A quantitative review and the future outlook 
Kyungyun Lee, Keunwoo Choi and Juhan Nam 

Review paper about singing voice detection. Common problems are identified where current methods fail: 1. Low SNR ratio between vocals and instrumentals 2. Guitar and other instruments that sound "similar" to voice are mistaken for it and 3. Presence of vibrato in other instruments is mistaken for singing voice. These findings give inspiration on how to improve future systems.

## (D-15) Vocals in Music Matter: the Relevance of Vocals in the Minds of Listeners 
Andrew Demetriou, Andreas Jansson, Aparna Kumar and Rachel Bittner 

Psychological qualtiative and quantitative studies demonstrate that listeners attend very closely to singing voice compared to many other aspects of music, despite the lack of singing voice related attributes in music tags for songs.

## (D-16) Vocal melody extraction with semantic segmentation and audio-symbolic domain transfer learning 
Wei Tsung Lu and Li Su 

For vocal melody extraction, a symbolic vocal segmentation model is first trained on symbolic data. Then the vocal melody extractor is trained from the audio plus the symbolic representation extracted by the other model (we assume audio and symbolic input is known for each sample). At test time, since symbolic data is not available, a simple filter is applied to the audio to get an estimate of what the symbolic transcription might look like to feed it into the symbolic model, before its output is fed into the audio model.

## (D-17) Empirically Weighting the Importance of Decision Factors for Singing Preference 
Michael Barone, Karim Ibrahim, Chitralekha Gupta and Ye Wang 

Psychological study into how important different factors (familiarity, genre preference,
ease of vocal reproducibility, and overall preference of the song) are for predicting how attractive it is for a person to sing along to a song.

# Timbre, tagging, similarity, patterns and alignment

## (E-3) Comparison of Audio Features for Recognition of Western and Ethnic Instruments in Polyphonic Mixtures 
Igor Vatolkin and Günter Rudolph 

Using evolutionary optimisation to select features most useful for detecting Western or Ethnic instruments. Since these feature sets turn out to be somewhat different, they also search for the best "compromise set" of features that performs reasonably well (but worse than the specialised features) on both types of data.

## (E-4) Instrudive: A Music Visualization System Based on Automatically Recognized Instrumentation 
Takumi Takahashi, Satoru Fukayama and Masataka Goto 

Visualising a collection of music pieces by turning each piece into a pie-chart that shows the percentage of time each instrument is active.

## (E-6) Jazz Solo Instrument Classification with Convolutional Neural Networks, Source Separation, and Transfer Learning 
Juan S. Gómez, Jakob Abeßer and Estefanía Cano 
 
To classify the particular jazz solo instrument playing, source separation is used to remove other instruments first, which helps classification performance. Transfer learning on the other hand by using a model trained on a different dataset beforehand does not turn out to work better, but that may be due to the way the model predictions are aggregated to compute the evaluation metrics.

## (E-9) Semi-supervised lyrics and solo-singing alignment 
Chitralekha Gupta, Rong Tong, Haizhou Li and Ye Wang 

Usage of the DAMP data containing amateur solo singing recordings together with unaligned lyrics, which are roughly aligned by using existing speech recognition technology, to train a lyrics transcription and alignment system. They reach a word error rate of 36%, however it is not known how much this degrades on normal music with lots of accopaniment noise.

## (E-14) End-to-end Learning for Music Audio Tagging at Scale 
Jordi Pons, Oriol Nieto, Matthew Prockup, Erik M. Schmidt, Andreas F. Ehmann and Xavier Serra 

Comparison of spectrogram-based with direct raw audio-based classification models for music tagging with varying sizes of training data indicates that spectrograms lead to slightly better performance for small, but slightly worse performance for very large training datasets compared to direct audio input.

## (E-17) Learning Interval Representations from Polyphonic Music Sequences 
Stefan Lattner, Maarten Grachten and Gerhard Widmer 

Instead of modeling a sequence of pitches directly, we model the transformation of previous pitches into the current one with a gated auto-encoder and then let the RNN model the autoencoder embeddings, which makes for key-invariant processing.

# Session F - Machine and human learning of music 

## (F-3) Listener Anonymizer: Camouflaging Play Logs to Preserve User’s Demographic Anonymity 
Kosetsu Tsukuda, Satoru Fukayama and Masataka Goto 
 
Individual users of music streaming services can protect themselves against being identified in terms of their nationality, age, etc. by way of their playlist history with this technique, which estimates these attributes internally and then tells the user which songs he should play to confuse the recommendation engine to obfuscate these attributes.

## (F-7) Representation Learning of Music Using Artist Labels 
Jiyoung Park, Jongpil Lee, Jangyeon Park, Jung-Woo Ha and Juhan Nam 

Instead of classifying genre directly, which limits training data and introduces label noise, train to detect the artist, which are objective, easily obtained labels, first. Then use the learned feature representation in the last layer to perform genre detection on a few different datasets

## (F-11) MIDI-VAE: Modeling Dynamics and Instrumentation of Music with Applications to Style Transfer 
Gino Brunner, Andres Konrad, Yuyi Wang and Roger Wattenhofer 

Use a VAE on short audio excerpts, but reserve 2 dimensions in the latent space for modeling different musical styles (jazz, classic, …) by ensuring that a classifier using these dimensions can identify the style of the input. Then the VAE can be used for style transfer by encoding a given input, and changing the style code in the latent space before decoding it.

## (F-12) Understanding a Deep Machine Listening Model Through Feature Inversion 
Saumitra Mishra, Bob L. Sturm and Simon Dixon 

To understand what information/concept is captured by each layer/neuron at a deep audio model, extra decoder functions are trained at each layer to recover the original input of the network (which gets harder the closer you get to the classification layer).

## (F-16) Learning to Listen, Read, and Follow: Score Following as a Reinforcement Learning Game 
Matthias Dorfer, Florian Henkel and Gerhard Widmer 

Apply reinforcement learning to score following by defining an agent that looks at the current section of music sheet and audio spectrogram and then decides whether to increase or decrease the current note sheet scrolling speed.