#!/usr/bin/env python3
"""
Recipe for training a Voice Activity Detection (VAD) model on LibriParty.
This code heavily relis on data augmentation with external datasets.
(e.g, open_rir, musan, CommonLanguage is used as well).

Make sure you download all the datasets before staring the experiment:
- LibriParty: https://www.dropbox.com/s/8zcn6zx4fnxvfyt/LibriParty.tar.gz?dl=0
- Musan: https://www.openslr.org/resources/17/musan.tar.gz
- CommonLanguage: https://zenodo.org/record/5036977/files/CommonLanguage.tar.gz?download=1

To run an experiment:

python train.py hparams/train.yaml\
--data_folder=/path/to/LibriParty \
--musan_folder=/path/to/musan/\
--commonlanguage_folder=/path/to/commonlang

Authors
 * Mohamed Kleit 2021
 * Arjun V 2021
 * Mirco Ravanelli 2021
"""

import sys

import numpy as np
import torch
import torchaudio
from data_augment import augment_data, EmbeddingAugmentation
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
from torch.utils.data import random_split

logger = get_logger(__name__)


# sb.utils.seed.seed_everything(402)


class VADBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Given an input batch it computes the binary probability.
        In training phase, we create on-the-fly augmentation data.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.signal
        targets, lens_targ = batch.target
        target_spkr_wavs, target_spkr_lens = batch.sample_signal
        self.targets = targets

        # print("wavs", wavs.shape)
        # print("targets", targets.shape)
        # print("lens", lens.shape)
        # print("lens targets", lens_targ.shape)

        if stage == sb.Stage.TRAIN:
            wavs, targets, lens = augment_data(
                self.noise_datasets,
                self.speech_datasets,
                wavs,
                targets,
                lens_targ,
            )
            self.lens = lens
            self.targets = targets

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        feats = feats.detach()
        outputs = self.modules.cnn(feats)
        # output1 = outputs

        # print("wavs", wavs.shape)
        # print("targets", targets.shape)
        # print("lens targets", lens_targ.shape)
        # print("feats", feats.shape)
        # print("outputs shape", outputs.shape)

        outputs = outputs.reshape(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] * outputs.shape[3],
        )
        # output2 = outputs

        embs = self.modules.verification.encode_batch(target_spkr_wavs, target_spkr_lens).detach()
        embs = embs.expand(embs.shape[0], outputs.shape[1], -1)

        if stage == sb.Stage.TRAIN:
            augmentor = EmbeddingAugmentation(
                noise_std=0.01, 
                dropout_prob=0.1, 
                salt_pepper_prob=0.01, 
                frequency_factor=0.1,
                noise_types=["gaussian", "dropout", "salt_pepper", "frequency"],
            )

            # Apply augmentation to multiple batches
            embs = augmentor.multi_forward(embs, out_batch_size=outputs.shape[0]//embs.shape[0])
        else:
            # Repeat the speaker embeddings to match the augmented data. The order is consistent with the augmented data. 
            embs = embs.repeat(outputs.shape[0]//embs.shape[0], 1, 1)
        
        outputs = torch.cat((outputs, embs), dim=-1)
        # output3 = outputs

        # print("outputs after cnn and reshape", outputs.shape)
        outputs, h = self.modules.rnn(outputs)
        # output4 = outputs
        # print("outputs after rnn", outputs.shape)
        outputs = self.modules.dnn(outputs)
        # output5 = outputs
        # if torch.isnan(outputs).any():
        #     print("NAN outputs")
        #     print("output1", output1)
        #     print("output2", output2)
        #     print("output3", output3)
        #     print("output4", output4)
        #     print("output5", output5)
        #     exit()
        # print("outputs after dnn", outputs.shape)
        # exit()
        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the binary CE"
        predictions, lens = predictions
        targets = self.targets

        predictions = predictions[:, : targets.shape[-1], 0]

        loss = self.hparams.compute_BCE_cost(predictions, targets, lens)
        # if torch.isnan(loss).item():
        #     print(loss.shape)
        #     loss[()] = 0
        #     # loss[0] = 0
        #     print("NAN loss")
        #     print("loss", loss)
        #     print(predictions)
        #     print(targets)
        #     print(lens)


        self.train_metrics.append(batch.id, torch.sigmoid(predictions), targets)
        if stage != sb.Stage.TRAIN:
            self.valid_metrics.append(
                batch.id, torch.sigmoid(predictions), targets
            )
        return loss

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        self.train_metrics = self.hparams.train_stats()

        self.noise_datasets = [
            self.hparams.add_noise,
            self.hparams.add_noise_musan,
            self.hparams.add_music_musan,
        ]
        self.speech_datasets = [
            self.hparams.add_speech_musan,
            self.hparams.add_speech_musan,
            self.hparams.add_speech_musan,
        ]

        if stage != sb.Stage.TRAIN:
            self.valid_metrics = self.hparams.test_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            summary = self.valid_metrics.summarize(threshold=0.5)

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "summary": summary},
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_loss, "summary": summary},
                num_to_keep=1,
                min_keys=["loss"],
                name="epoch_{}".format(epoch),
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "summary": summary},
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    # 1. Declarations:
    train = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["annotation_train"],
    )
    validation = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["annotation_valid"],
    )
    test = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["annotation_test"],
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("signal")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig
    
    @sb.utils.data_pipeline.takes("target_speaker")
    @sb.utils.data_pipeline.provides("sample_signal")
    def target_audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav["sample"])
        return sig

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("target_speech")
    @sb.utils.data_pipeline.provides("target")
    def vad_targets(speech, hparams=hparams):
        boundaries = (
            [
                (
                    int(interval[0] / hparams["time_resolution"]),
                    int(interval[1] / hparams["time_resolution"]),
                )
                for interval in speech
            ]
            if len(speech) > 0
            else []
        )
        gt = torch.zeros(
            int(
                np.ceil(
                    hparams["example_length"] * (1 / hparams["time_resolution"])
                )
            )
        )
        for indxs in boundaries:
            start, stop = indxs
            gt[start:stop] = 1
        return gt

    # Create dataset
    datasets = [train, validation, test]
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, target_audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, vad_targets)
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "signal", "target", "sample_signal"]
    )

    # Split dataset
    train_data, valid_data, test_data = datasets
    return train_data, valid_data, test_data


# Begin Recipe!
if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # TODO: merge this from ami_prepare.py
    from ami_prepare import prepare_ami

    # LibriParty preparation
    if not hparams["skip_prep"]:
        run_on_main(
            prepare_ami,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["save_folder"],
                "ref_rttm_dir": hparams["ref_rttm_dir"],
                "meta_data_dir": hparams["meta_data_dir"],
                "manual_annot_folder": hparams["manual_annot_folder"],
                "split_type": hparams["split_type"],
                "skip_TNO": hparams["skip_TNO"],
                "mic_type": hparams["mic_type"],
                "max_subseg_dur": hparams["max_subseg_dur"],
                "subseg_overlap": hparams["subseg_overlap"],
            },
        )

    # Prepare openrir
    run_on_main(hparams["prepare_noise_data"])

    # Prepare Musan
    from musan_prepare import prepare_musan

    if not hparams["skip_prep"]:
        run_on_main(
            prepare_musan,
            kwargs={
                "folder": hparams["musan_folder"],
                "music_csv": hparams["music_csv"],
                "noise_csv": hparams["noise_csv"],
                "speech_csv": hparams["speech_csv"],
                "max_noise_len": hparams["example_length"],
            },
        )

    # Prepare common
    from commonlanguage_prepare import prepare_commonlanguage

    if not hparams["skip_prep"]:
        run_on_main(
            prepare_commonlanguage,
            kwargs={
                "folder": hparams["commonlanguage_folder"],
                "csv_file": hparams["multilang_speech_csv"],
            },
        )

    # Dataset IO prep: creating Dataset objects
    train_data, valid_data, test_data = dataio_prep(hparams)

    # Train only on a subset of the data
    # print("before", train_data.data_ids[:10])
    if hparams["fast_train"]:
        # train_data = train_data.filtered_sorted(select_n=hparams["max_train_data"])
        # valid_data = valid_data.filtered_sorted(select_n=hparams["max_valid_data"])
        # test_data = test_data.filtered_sorted(select_n=hparams["max_test_data"])

        # train_data = torch.utils.data.Subset(train_data, range(hparams["max_train_data"]))
        # valid_data = torch.utils.data.Subset(valid_data, range(hparams["max_valid_data"]))
        # test_data = torch.utils.data.Subset(test_data, range(hparams["max_test_data"]))

        train_data.data_ids = train_data.data_ids[:hparams["max_train_data"]]
        valid_data.data_ids = valid_data.data_ids[:hparams["max_valid_data"]]
        test_data.data_ids = test_data.data_ids[:hparams["max_test_data"]]
    # print("after", train_data.data_ids[:10])
    
    print("train_data", len(train_data))
    print("valid_data", len(valid_data))
    print("test_data", len(test_data))

    verification = SpeakerRecognition.from_hparams(hparams["ecapa_pretrain_path"], savedir=hparams["ecapa_save_path"], run_opts={"device": "cuda"})
    hparams["modules"].update({"verification": verification})

    # Trainer initialization
    vad_brain = VADBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training/validation loop
    # with torch.autograd.detect_anomaly():
    vad_brain.fit(
        vad_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
        progressbar=True,
    )

    # Test
    vad_brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
