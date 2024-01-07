import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.models import ConvTasNet
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm
import wandb
import numpy as np
import copy 
from getmodel import get_model
from pystoi import stoi
from pesq import pesq

class Trainer:
    def __init__(
        self,
        train_data,
        val_data,
        checkpoint_name,
        display_freq=10,
        useWandB = False,
        noisySample = None,
        cleanSample = None,
        sample_rate = 16000,
        duration = 4,
    ):
        self.train_data = train_data
        self.val_data = val_data
        assert checkpoint_name.endswith(".tar"), "The checkpoint file must have .tar extension"
        self.checkpoint_name = checkpoint_name
        self.display_freq = display_freq
        self.useWandB = useWandB
        self.modelCopy = None
        self.noisySample = noisySample
        self.cleanSample = cleanSample
        self.sample_rate = sample_rate
        self.duration = duration

    def fit(
        self,
        model,
        device,
        epochs=10,
        batch_size=16,
        lr=0.001,
        weight_decay=1e-5,
        optimizer=optim.Adam,
        loss_fn=F.mse_loss,
        loss_mode="min",
        gradient_clipping=True,
    ):
        
        initial_pesq_score = pesq(fs = self.sample_rate,ref = self.cleanSample,deg = np.squeeze(self.noisySample))
        initial_estoi_score = stoi(x=self.cleanSample,y = np.squeeze(self.noisySample),fs_sig= self.sample_rate,extended=True)
        print(f"Initial PESQ Score: {initial_pesq_score}")
        print(f"Initial ESTOI Score: {initial_estoi_score}")

        if(self.useWandB):
              print("Using WandB")
              wandb.init(
                        # set the wandb project where this run will be logged
                        project="ConvTasnetImpl",
                        name= f'lr:{lr}_bs:{batch_size}',
                        config={
                            "epochs":epochs,
                            "learning_rate":lr ,
                            "batch_size": batch_size,
                        }
                    )
              wandb.log({"model trainable parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                         'trainCleanAudio': wandb.Audio(self.cleanSample, caption="Clean Speech", sample_rate= self.sample_rate),
                         'trainNoisyAudio': wandb.Audio(np.squeeze(self.noisySample), caption="Noisy Speech", sample_rate= self.sample_rate),
                         'initial_pesq_score': initial_pesq_score,
                         'initial_estoi_score': initial_estoi_score 
                         })
            

        # Get the device placement and make data loaders
        self.device = device
        print(f"Using device: {self.device}")
        kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size,generator = torch.Generator(device=device),**kwargs)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size,generator = torch.Generator(device=device), **kwargs)

        self.optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = loss_fn
        self.loss_mode = loss_mode
        self.gradient_clipping = gradient_clipping
        self.history = {"train_loss": [], "test_loss": []}

        previous_epochs = 0
        best_loss = None

        # Try loading checkpoint (if it exists)
        if os.path.isfile(self.checkpoint_name):
            print(f"Resuming training from checkpoint: {self.checkpoint_name}")
            checkpoint = torch.load(self.checkpoint_name)
            model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.loss_fn = checkpoint["loss_fn"]
            self.history = checkpoint["history"]
            previous_epochs = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
        else:
            print(f"No checkpoint found, using default parameters...")

        for epoch in range(previous_epochs + 1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}:")
            train_loss = self.train(model)
            test_loss = self.test(model)

            if(self.useWandB):
                wandb.log({"train_loss": train_loss, "test_loss": test_loss},step = epoch)
            
            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)

            # Save checkpoint only if the validation loss improves (avoid overfitting)
            if (
                best_loss is None
                or (test_loss < best_loss and self.loss_mode == "min")
                or (test_loss > best_loss and self.loss_mode == "max")
            ):
                print(f"Validation loss improved from {best_loss} to {test_loss}.")
                print(f"Saving checkpoint to: {self.checkpoint_name}")
                best_loss = test_loss

                checkpoint_data = {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss_fn": self.loss_fn,
                    "history": self.history,
                }
                torch.save(checkpoint_data, f'{self.checkpoint_name}')

            if(epoch % 2 == 0):
                print("Making inference during training.....")
                # Make a copy of model for in training inferences
                self.modelCopy = copy.deepcopy(model)
                self.modelCopy.eval()
                
                with torch.no_grad():
                    total_samples = self.noisySample.shape[1]
                    segment_length = self.sample_rate * self.duration
                    n_segments = int(np.ceil(self.noisySample.shape[1] / segment_length))

                    output_segments = {"clean": []}
                    for i in range(n_segments):
                        # print(f"Processing segment {i+1}/{n_segments}")
                        if self.noisySample.shape[1] >= (i + 1) * segment_length:
                            seg_audio = self.noisySample[:, i * segment_length : (i + 1) * segment_length]
                        else:
                            seg_audio = torch.zeros([1, segment_length])
                            seg_audio[:, 0 : self.noisySample.shape[1] - i * segment_length] = self.noisySample[:, i * segment_length :]
                        
                    
                        seg_audio = torch.from_numpy(seg_audio).unsqueeze(0).to(self.device)
                        print(f'seg_audio shape = {seg_audio.shape}')
                        out_sources = self.modelCopy(seg_audio)  # Use the model
    
                        out_sources = out_sources.squeeze()
                        out_sources = out_sources.cpu().detach()

                        clean_audio = out_sources[0:1, :]

                        #Normalize
                        clean_audio /= clean_audio.abs().max()
                        # Append the obtained segments for each source into a list
                        output_segments["clean"].append(clean_audio)

                    # Concatenate along time dimension to obtain the full audio
                    clean_output = torch.cat(output_segments["clean"], dim=1)
                    clean_output = clean_output.cpu().detach().numpy()

                    pesq_score = pesq(fs = self.sample_rate,ref = self.cleanSample,deg = np.squeeze(clean_output))
                    estoi_score = stoi(x=self.cleanSample,y = np.squeeze(clean_output),fs_sig= self.sample_rate,extended=True)
                    
                    print(f"Epoch: {epoch} PESQ Score: {pesq_score}")
                    print(f"Epoch: {epoch} ESTOI Score: {estoi_score}")

                    if(self.useWandB):
                        wandb.log({'PESQ Score': pesq_score, 
                                   'ESTOI Score': estoi_score,
                                   'DenoisedAudio': wandb.Audio(np.squeeze(clean_output), caption="Denoised Speech", sample_rate= self.sample_rate),
                                   }
                                   ,step = epoch)

        if(self.useWandB):
            wandb.finish()

        return self.history

    def train(self, model):
        total_loss = 0.0
        model.train()
        with tqdm(self.train_loader) as progress:
            for i, (mixture, sources) in enumerate(progress):
                mixture = mixture.to(self.device)
                sources = sources.to(self.device)

                self.optimizer.zero_grad()

                predictions = model(mixture)
                loss = self.loss_fn(predictions, sources)

                if self.loss_mode == "max":  # To optimize for maximization, multiply by -1
                    loss = -1 * loss

                loss.mean().backward()

                # Gradient Value Clipping
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

                self.optimizer.step()

                total_loss += loss.mean().item()

                if i % self.display_freq == 0:
                    progress.set_postfix(
                        {
                            "loss": float(total_loss / (i + 1)),
                        }
                    )

        total_loss /= len(self.train_loader)
        return total_loss

    def test(self, model):
        total_loss = 0.0
        model.eval()
        with torch.no_grad():
            with tqdm(self.val_loader) as progress:
                for i, (mixture, sources) in enumerate(progress):
                    mixture = mixture.to(self.device)
                    sources = sources.to(self.device)

                    predictions = model(mixture)

                    loss = self.loss_fn(predictions, sources)

                    total_loss += loss.mean().item()

                    if i % self.display_freq == 0:
                        progress.set_postfix(
                            {
                                "loss": float(total_loss / (i + 1)),
                            }
                        )

        total_loss /= len(self.val_loader)
        return total_loss


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Datasets
    ap.add_argument("--clean_train_path",default= "datasets/Train/cleanSliced" )
    ap.add_argument("--clean_val_path",default = "datasets/Validation/cleanSliced")
    ap.add_argument("--noise_train_path",default = "datasets/Train/noisySliced")
    ap.add_argument("--noise_val_path",default = "datasets/Validation/noisySliced")
    ap.add_argument("--keep_rate", default=1, type=float)

    # Model checkpoint
    ap.add_argument("--model",default="ConvTasNet", choices=["UNet", "UNetDNP", "ConvTasNet", "TransUNet", "SepFormer"])
    ap.add_argument("--checkpoint_name", default= "LearningRate0001.tar",help="File with .tar extension")

    # Training params
    ap.add_argument("--epochs", default=50, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--gradient_clipping", action="store_true")

    # GPU setup
    # ap.add_argument("--gpu", default="-1")
    # ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # visible_devices = list(map(lambda x: int(x), args.gpu.split(",")))
    # print("Visible devices:", visible_devices)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
    print(f"Using device: {device}")

    # from torchaudio.models import ConvTasNet

    # from losses import LogSTFTMagnitudeLoss, MultiResolutionSTFTLoss, ScaleInvariantSDRLoss, SpectralConvergenceLoss
    # from models import *

    # Select the model to be used for training
    training_utils_dict = get_model(args.model)

    model = training_utils_dict["model"]
    data_mode = training_utils_dict["data_mode"]
    loss_fn = training_utils_dict["loss_fn"]
    loss_mode = training_utils_dict["loss_mode"]

    # model = torch.nn.DataParallel(model, device_ids=list(range(len(visible_devices))))
    model = model.to(device)

    from data import AudioDirectoryDataset, NoiseMixerDataset

    train_data = NoiseMixerDataset(
        clean_dataset=AudioDirectoryDataset(root=args.clean_train_path, keep_rate=args.keep_rate),
        noise_dataset=AudioDirectoryDataset(root=args.noise_train_path, keep_rate=args.keep_rate),
        mode=data_mode,
    )

    val_data = NoiseMixerDataset(
        clean_dataset=AudioDirectoryDataset(root=args.clean_val_path, keep_rate=args.keep_rate),
        noise_dataset=AudioDirectoryDataset(root=args.noise_val_path, keep_rate=args.keep_rate),
        mode=data_mode,
    )

    print(f"Train data: {len(train_data)} samples")
    print(f"Val data: {len(val_data)} samples")

    trainingSample = []
    cleanSample = []
    for i,(mixture, sources) in enumerate(train_data):
        if(i == 0):
            trainingSample = np.array(mixture)
            cleanSample = np.array(sources[0])
        elif i == 3:
            break
        else:
            trainingSample = np.concatenate([trainingSample,mixture],axis=1)
            cleanSample = np.concatenate([cleanSample,sources[0]],axis=0)


    useWandB = False

    # trainer = Trainer(train_data, val_data, checkpoint_name=args.checkpoint_name,useWandB = useWandB,noisySample =trainingSample,cleanSample = cleanSample,sample_rate= 16000, duration= 4)
    # history = trainer.fit(
    #     model,
    #     device,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     lr=args.lr,
    #     loss_fn=loss_fn,
    #     loss_mode=loss_mode,
    #     gradient_clipping=args.gradient_clipping,
    # )
