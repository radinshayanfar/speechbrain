import torch
import torchaudio

import speechbrain as sb
from speechbrain.inference.VAD import VAD
from speechbrain.inference.interfaces import Pretrained
from speechbrain.utils.data_utils import split_path
from speechbrain.utils.fetching import fetch
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.DER import DER
import tqdm
import os
import json
import pytest

# pytest.skip('Skipping because of Perl dependency')

class SVAD(VAD):
    def get_speech_prob_file(
        self,
        audio_file,
        large_chunk_size=30,
        small_chunk_size=10,
        overlap_small_chunk=False,
        target_spkr_emb=None,
    ):
        """Outputs the frame-level speech probability of the input audio file
        using the neural model specified in the hparam file. To make this code
        both parallelizable and scalable to long sequences, it uses a
        double-windowing approach.  First, we sequentially read non-overlapping
        large chunks of the input signal.  We then split the large chunks into
        smaller chunks and we process them in parallel.

        Arguments
        ---------
        audio_file: path
            Path of the audio file containing the recording. The file is read
            with torchaudio.
        large_chunk_size: float
            Size (in seconds) of the large chunks that are read sequentially
            from the input audio file.
        small_chunk_size: float
            Size (in seconds) of the small chunks extracted from the large ones.
            The audio signal is processed in parallel within the small chunks.
            Note that large_chunk_size/small_chunk_size must be an integer.
        overlap_small_chunk: bool
            True, creates overlapped small chunks. The probabilities of the
            overlapped chunks are combined using hamming windows.

        Returns
        -------
        prob_vad: torch.Tensor
            torch.Tensor containing the frame-level speech probabilities for the
            input audio file.
        """
        # Getting the total size of the input file
        sample_rate, audio_len = self._get_audio_info(audio_file)

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        # Computing the length (in samples) of the large and small chunks
        long_chunk_len = int(sample_rate * large_chunk_size)
        small_chunk_len = int(sample_rate * small_chunk_size)

        # Setting the step size of the small chunk (50% overlapping windows are supported)
        small_chunk_step = small_chunk_size
        if overlap_small_chunk:
            small_chunk_step = small_chunk_size / 2

        # Computing the length (in sample) of the small_chunk step size
        small_chunk_len_step = int(sample_rate * small_chunk_step)

        # Loop over big chunks
        prob_chunks = []
        last_chunk = False
        begin_sample = 0
        while True:
            # Check if the current chunk is the last one
            if begin_sample + long_chunk_len >= audio_len:
                last_chunk = True

            # Reading the big chunk
            large_chunk, fs = torchaudio.load(
                str(audio_file),
                frame_offset=begin_sample,
                num_frames=long_chunk_len,
            )
            large_chunk = large_chunk.to(self.device)

            # Manage padding of the last small chunk
            if last_chunk or large_chunk.shape[-1] < small_chunk_len:
                padding = torch.zeros(
                    1, small_chunk_len, device=large_chunk.device
                )
                large_chunk = torch.cat([large_chunk, padding], dim=1)

            # Splitting the big chunk into smaller (overlapped) ones
            small_chunks = torch.nn.functional.unfold(
                large_chunk.unsqueeze(1).unsqueeze(2),
                kernel_size=(1, small_chunk_len),
                stride=(1, small_chunk_len_step),
            )
            small_chunks = small_chunks.squeeze(0).transpose(0, 1)

            # Getting (in parallel) the frame-level speech probabilities
            small_chunks_prob = self.get_speech_prob_chunk(small_chunks, target_spkr_emb=target_spkr_emb)
            small_chunks_prob = small_chunks_prob[:, :-1, :]

            # Manage overlapping chunks
            if overlap_small_chunk:
                small_chunks_prob = self._manage_overlapped_chunks(
                    small_chunks_prob
                )

            # Prepare for folding
            small_chunks_prob = small_chunks_prob.permute(2, 1, 0)

            # Computing lengths in samples
            out_len = int(
                large_chunk.shape[-1] / (sample_rate * self.time_resolution)
            )
            kernel_len = int(small_chunk_size / self.time_resolution)
            step_len = int(small_chunk_step / self.time_resolution)

            # Folding the frame-level predictions
            small_chunks_prob = torch.nn.functional.fold(
                small_chunks_prob,
                output_size=(1, out_len),
                kernel_size=(1, kernel_len),
                stride=(1, step_len),
            )

            # Appending the frame-level speech probabilities of the large chunk
            small_chunks_prob = small_chunks_prob.squeeze(1).transpose(-1, -2)
            prob_chunks.append(small_chunks_prob)

            # Check stop condition
            if last_chunk:
                break

            # Update counter to process the next big chunk
            begin_sample = begin_sample + long_chunk_len

        # Converting the list to a tensor
        prob_vad = torch.cat(prob_chunks, dim=1)
        last_elem = int(audio_len / (self.time_resolution * sample_rate))
        prob_vad = prob_vad[:, 0:last_elem, :]

        return prob_vad

    def get_speech_prob_chunk(self, wavs, wav_lens=None, target_spkr_emb=None):
        """Outputs the frame-level posterior probability for the input audio chunks
        Outputs close to zero refers to time steps with a low probability of speech
        activity, while outputs closer to one likely contain speech.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        outputs = self.mods.cnn(feats)

        outputs = outputs.reshape(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] * outputs.shape[3],
        )

        embs = target_spkr_emb.expand(outputs.shape[0], outputs.shape[1], -1)
        # print(embs.shape)
        # print(outputs.shape)
        outputs = torch.cat((outputs, embs), dim=-1)

        outputs, h = self.mods.rnn(outputs)
        outputs = self.mods.dnn(outputs)
        output_prob = torch.sigmoid(outputs)

        return output_prob

    def double_check_speech_segments(
        self, boundaries, audio_file, speech_th=0.5, target_spkr_emb=None,
    ):
        """Takes in input the boundaries of the detected speech segments and
        double checks (using the neural VAD) that they actually contain speech.

        Arguments
        ---------
        boundaries: torch.Tensor
            torch.Tensor containing the boundaries of the speech segments.
        audio_file: path
            The original audio file used to compute vad_out.
        speech_th: float
            Threshold on the mean posterior probability over which speech is
            confirmed. Below that threshold, the segment is re-assigned to a
            non-speech region.

        Returns
        -------
        new_boundaries
            The boundaries of the segments where speech activity is confirmed.
        """

        # Getting the total size of the input file
        sample_rate, sig_len = self._get_audio_info(audio_file)

        # Double check the segments
        new_boundaries = []
        for i in range(boundaries.shape[0]):
            beg_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            len_seg = end_sample - beg_sample

            # Read the candidate speech segment
            segment, fs = torchaudio.load(
                str(audio_file), frame_offset=beg_sample, num_frames=len_seg
            )
            speech_prob = self.get_speech_prob_chunk(segment, target_spkr_emb=target_spkr_emb)
            if speech_prob.mean() > speech_th:
                # Accept this as a speech segment
                new_boundaries.append([boundaries[i, 0], boundaries[i, 1]])

        # Convert boundaries from list to tensor
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries
 
    def get_speech_segments(
        self,
        audio_file,
        large_chunk_size=30,
        small_chunk_size=10,
        overlap_small_chunk=False,
        apply_energy_VAD=False,
        double_check=True,
        close_th=0.250,
        len_th=0.250,
        activation_th=0.5,
        deactivation_th=0.25,
        en_activation_th=0.5,
        en_deactivation_th=0.0,
        speech_th=0.50,
        target_spkr_emb=None,
    ):
        """Detects speech segments within the input file. The input signal can
        be both a short or a long recording. The function computes the
        posterior probabilities on large chunks (e.g, 30 sec), that are read
        sequentially (to avoid storing big signals in memory).
        Each large chunk is, in turn, split into smaller chunks (e.g, 10 seconds)
        that are processed in parallel. The pipeline for detecting the speech
        segments is the following:
            1- Compute posteriors probabilities at the frame level.
            2- Apply a threshold on the posterior probability.
            3- Derive candidate speech segments on top of that.
            4- Apply energy VAD within each candidate segment (optional).
            5- Merge segments that are too close.
            6- Remove segments that are too short.
            7- Double check speech segments (optional).

        Arguments
        ---------
        audio_file : str
            Path to audio file.
        large_chunk_size: float
            Size (in seconds) of the large chunks that are read sequentially
            from the input audio file.
        small_chunk_size: float
            Size (in seconds) of the small chunks extracted from the large ones.
            The audio signal is processed in parallel within the small chunks.
            Note that large_chunk_size/small_chunk_size must be an integer.
        overlap_small_chunk: bool
            If True, it creates overlapped small chunks (with 50% overlap).
            The probabilities of the overlapped chunks are combined using
            hamming windows.
        apply_energy_VAD: bool
            If True, a energy-based VAD is used on the detected speech segments.
            The neural network VAD often creates longer segments and tends to
            merge close segments together. The energy VAD post-processes can be
            useful for having a fine-grained voice activity detection.
            The energy thresholds is  managed by activation_th and
            deactivation_th (see below).
        double_check: bool
            If True, double checks (using the neural VAD) that the candidate
            speech segments actually contain speech. A threshold on the mean
            posterior probabilities provided by the neural network is applied
            based on the speech_th parameter (see below).
        close_th: float
            If the distance between boundaries is smaller than close_th, the
            segments will be merged.
        len_th: float
            If the length of the segment is smaller than close_th, the segments
            will be merged.
        activation_th:  float
            Threshold of the neural posteriors above which starting a speech segment.
        deactivation_th: float
            Threshold of the neural posteriors below which ending a speech segment.
        en_activation_th: float
            A new speech segment is started it the energy is above activation_th.
            This is active only if apply_energy_VAD is True.
        en_deactivation_th: float
            The segment is considered ended when the energy is <= deactivation_th.
            This is active only if apply_energy_VAD is True.
        speech_th: float
            Threshold on the mean posterior probability within the candidate
            speech segment. Below that threshold, the segment is re-assigned to
            a non-speech region. This is active only if double_check is True.

        Returns
        -------
        boundaries: torch.Tensor
            torch.Tensor containing the start second of speech segments in even
            positions and their corresponding end in odd positions
            (e.g, [1.0, 1.5, 5,.0 6.0] means that we have two speech segment;
             one from 1.0 to 1.5 seconds and another from 5.0 to 6.0 seconds).
        """

        # Fetch audio file from web if not local
        source, fl = split_path(audio_file)
        audio_file = fetch(fl, source=source)

        # Computing speech vs non speech probabilities
        prob_chunks = self.get_speech_prob_file(
            audio_file,
            large_chunk_size=large_chunk_size,
            small_chunk_size=small_chunk_size,
            overlap_small_chunk=overlap_small_chunk,
            target_spkr_emb=target_spkr_emb,
        )

        # Apply a threshold to get candidate speech segments
        prob_th = self.apply_threshold(
            prob_chunks,
            activation_th=activation_th,
            deactivation_th=deactivation_th,
        ).float()

        # Compute the boundaries of the speech segments
        boundaries = self.get_boundaries(prob_th, output_value="seconds")

        # Apply energy-based VAD on the detected speech segments
        if apply_energy_VAD:
            boundaries = self.energy_VAD(
                audio_file,
                boundaries,
                activation_th=en_activation_th,
                deactivation_th=en_deactivation_th,
            )

        # Merge short segments
        boundaries = self.merge_close_segments(boundaries, close_th=close_th)

        # Remove short segments
        boundaries = self.remove_short_segments(boundaries, len_th=len_th)

        # Double check speech segments
        if double_check:
            boundaries = self.double_check_speech_segments(
                boundaries, audio_file, speech_th=speech_th, target_spkr_emb=target_spkr_emb,
            )

        return boundaries

    def forward(self, wavs, wav_lens=None, target_spkr_emb=None):
        """Gets frame-level speech-activity predictions"""
        return self.get_speech_prob_chunk(wavs, wav_lens, target_spkr_emb=target_spkr_emb)


def find_random_spkr_segment(file_id, speaker_id):
    # get random speaker segment
    with open('results/SVAD/1986/save/metadata/ami_eval.Mix-Lapel.vad_segs.json', 'r') as f:
        metadata = json.load(f)

    for key, value in metadata.items():
        if key.startswith(file_id) and value['target_speaker']['id'] == speaker_id:
            return value['target_speaker']['sample']['start'], value['target_speaker']['sample']['stop']
    return None, None

def create_prediction_rttm():
    # load the model checkpoint
    VAD = SVAD.from_hparams(source="results/SVAD/1986/save/CKPT+epoch_8", run_opts={"device": "cuda"})
    print("Checkpoint loaded")

    # load ecapa_tdnn model
    verification_model = SpeakerRecognition.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={"device": "cuda"})
    print("Speaker recognition model loaded")

    # infere model and get speech segments on evaluation data
    eval_files = ['ES2004a', 'ES2004b', 'ES2004c', 'ES2004d', 'IS1009a', 'IS1009b', 'IS1009c', 'IS1009d', 'EN2002a', 'EN2002b','EN2002c', 'EN2002d']
    print("evaluating on files: ", eval_files)

    with open(f"results/SVAD/1986/save/sys.rttm", 'w') as f:
        for file in tqdm.tqdm(eval_files):
            
            for speaker_id in [f'{file}.A' , f'{file}.B', f'{file}.C', f'{file}.D']:
                f.write(f"SPKR-INFO {file} 0 <NA> <NA> <NA> unknown {speaker_id} <NA> <NA>\n")
            
            speech_file = f"datasets/amicorpus/{file}/audio/{file}.Mix-Lapel.wav"

            rttm_lines = []
            
            for speaker_id in [f'{file}.A' , f'{file}.B', f'{file}.C', f'{file}.D']:
                start, stop = find_random_spkr_segment(file, speaker_id)
                if (start is None) or (stop is None):
                    print(f"No segment found for speaker {speaker_id}")
                    continue

                sig = sb.dataio.dataio.read_audio({'file': speech_file, 'start': start, 'stop': stop})

                if len(sig.shape) > 1:
                    sig = sig[:, 0]
                sig = sig.unsqueeze(0).to("cuda")
                target_spkr_emb = verification_model.encode_batch(sig).detach()

                boundaries = VAD.get_speech_segments(speech_file, target_spkr_emb=target_spkr_emb)
                
                # save the boundaries in rttm format
                for i in range(boundaries.shape[0]):
                    rttm_lines.append(f"SPEAKER {file} 0 {boundaries[i, 0]:.3f} {boundaries[i, 1]-boundaries[i, 0]:.3f} <NA> <NA> {speaker_id} <NA> <NA>")
            
            # sort rttm lines
            rttm_lines.sort(key=lambda x: float(x.split()[3]))
            for line in rttm_lines:
                f.write(line + '\n')

    print(f"RTTM file created")
    

if __name__ == "__main__":
    sys_rttm = "results/SVAD/1986/save/sys.rttm"
    ref_rttm = "results/SVAD/1986/save/ref_rttms/fullref_ami_eval.rttm"

    if not os.path.exists(sys_rttm):
        create_prediction_rttm()

    Scores = DER(ref_rttm, sys_rttm, ignore_overlap=True, collar=0.25)

    print(Scores)
    with open("results/SVAD/1986/save/scores.json", 'w') as f:
        json.dump(Scores, f)