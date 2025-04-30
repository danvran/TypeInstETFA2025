import pandas as pd
import glob
import os
#import csv
from itertools import product
from argparse import Namespace
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import CSVLogger
import logging
import sys

# <functions>
def find_global_min_max(folder_path):
    # Initialize dictionaries to store the global min and max values for each column
    global_min = {}
    global_max = {}
    
    # List all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        
        # Iterate through each column in the DataFrame
        for column in df.columns:
            if column not in global_min:
                global_min[column] = df[column].min()
                global_max[column] = df[column].max()
            else:
                global_min[column] = min(global_min[column], df[column].min())
                global_max[column] = max(global_max[column], df[column].max())
    
    return global_min, global_max

def compute_absolute_max(global_min, global_max):
    absolute_max = {}
    for column in global_min:
        absolute_max[column] = max(abs(global_min[column]), abs(global_max[column]))
    return absolute_max

def concatenate_csv_files(folder_path, output_file, small=False, hole=False):
    # List to store dataframes
    dataframes = []
    
    # List all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    if hole:
        substrings_to_remove = ["_0_OK",
                                "_4_OK",
                                "_8_OK",
                                "_12_OK",
                                "_16_OK",
                                "_20_OK",
                                "all"]
        filtered_csv_files = [string for string in csv_files if not any(sub in string for sub in substrings_to_remove)]

    
    # Read and append each CSV file
    for file in filtered_csv_files:
        print(file)
        df = pd.read_csv(file)
        # If small is True, take only the first 10% of the data
        if small:
            df = df.iloc[:int(len(df) * 0.1)]
        dataframes.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)
    
    print(f"Combined CSV saved to {output_file}")

def load_data(file_path: str, window_size: int, step_size: int, voltage_scaler, current_scaler) -> torch.tensor:
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(file_path)
    df['V_T1_I1'] = df['V_T1_I1'] / voltage_scaler
    df['I_T1_I1'] = df['I_T1_I1'] / current_scaler
    #print("Original data shape:", df.shape)

    # Convert the DataFrame to a PyTorch tensor
    data_tensor = torch.tensor(df.values, dtype=torch.float32)

    # Create slices using a sliding window
    num_windows = (data_tensor.size(0) - window_size) // step_size + 1
    windows = [data_tensor[i*step_size:i*step_size+window_size] for i in range(num_windows)]
    sliced_data = torch.stack(windows)

    return sliced_data
# <functions>

# <classes>
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

class LitAutoEncoder(L.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        self.latent_dim = self.params.hidden_size
        self.encoder = nn.Sequential(nn.Linear(200, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, self.latent_dim))
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, 200))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch
        x = x.view(-1, 200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        #x_hat = x_hat.view(batchsize, 100, 2)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch
        x = x.view(-1, 200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log('epoch', torch.tensor(self.current_epoch, dtype=torch.float32), prog_bar=False)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)

    def threshold_step(self, batch, batch_idx):
        x = batch
        x = x.view(-1, 200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log('threshold_loss', val_loss, on_step=False, on_epoch=True)

    def forward(self, sample):
        x = sample.view(200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        return loss, sample, x_hat.view(100, 2)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
class LitDenoisingAutoEncoder(L.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.latent_dim = self.params.hidden_size
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, self.latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 50), nn.ReLU(),
            nn.Linear(50, 100), nn.ReLU(),
            nn.Linear(100, 200)
        )
        self.noise_std = 0.1  # You can tweak this

    def add_noise(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x

    def training_step(self, batch, batch_idx):
        x = batch.view(-1, 200)
        x_noisy = self.add_noise(x)
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.view(-1, 200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log('epoch', torch.tensor(self.current_epoch, dtype=torch.float32), prog_bar=False)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)

    def threshold_step(self, batch, batch_idx):
        x = batch.view(-1, 200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log('threshold_loss', val_loss, on_step=False, on_epoch=True)

    def forward(self, sample):
        x = sample.view(200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        return loss, sample, x_hat.view(100, 2)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

class LitVariationalAutoEncoder(L.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.latent_dim = self.params.hidden_size
        self.save_hyperparameters()
        
        # Encoder: outputs mu and logvar
        self.encoder = nn.Sequential(
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU()
        )
        self.fc_mu = nn.Linear(50, self.latent_dim)
        self.fc_logvar = nn.Linear(50, self.latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 50), nn.ReLU(),
            nn.Linear(50, 100), nn.ReLU(),
            nn.Linear(100, 200)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_loss(self, x, x_hat, mu, logvar):
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl_div, recon_loss, kl_div

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def training_step(self, batch, batch_idx):
        x = batch.view(-1, 200)
        z, mu, logvar = self.encode(x)
        x_hat = self.decoder(z)
        loss, recon_loss, kl = self.compute_loss(x, x_hat, mu, logvar)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.view(-1, 200)
        z, mu, logvar = self.encode(x)
        x_hat = self.decoder(z)
        val_loss, _, _ = self.compute_loss(x, x_hat, mu, logvar)
        self.log('epoch', torch.tensor(self.current_epoch, dtype=torch.float32), prog_bar=False)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)

    def threshold_step(self, batch, batch_idx):
        x = batch.view(-1, 200)
        z, mu, logvar = self.encode(x)
        x_hat = self.decoder(z)
        loss, _, _ = self.compute_loss(x, x_hat, mu, logvar)
        self.log('threshold_loss', loss, on_step=False, on_epoch=True)

    def forward(self, sample):
        x = sample.view(200)
        z, _, _ = self.encode(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        return loss, sample, x_hat.view(100, 2)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

# <classes>

# <Callbacks>
class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
    
class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Access the logged train loss
        train_loss = trainer.callback_metrics.get('train_loss')
        if train_loss is not None:
            self.train_loss.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Access the logged validation loss
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.val_loss.append(val_loss.item())    

# <Callbacks>

def main(path: str):
    R = True
    C = True
    I = True
    index = path[-1]
    if not os.path.exists(f'{path}/train/RLC_T{index}_all.csv'):
        print('Creating file with merged data')
        concatenate_csv_files(f'{path}/train', f'{path}/train/RLC_T{index}_all.csv')
    
    if not os.path.exists(f'{path}/train/RLC_T{index}_all_small.csv'):
        print('Creating file with merged data')
        concatenate_csv_files(f'{path}/train', f'{path}/train/RLC_T{index}_all_small.csv', small=True)
    
    if not os.path.exists(f'{path}/train/RLC_T{index}_all_holes.csv'):
        print('Creating file with merged data')
        concatenate_csv_files(f'{path}/train', f'{path}/train/RLC_T{index}_all_holes.csv', hole=True)

    # Get Min Max Values for global scaling
    folder_path = f'{path}/train'
    min_values, max_values = find_global_min_max(folder_path)
    print("Global Min Values:", min_values)
    print("Global Max Values:", max_values)

    absolute_max_values = compute_absolute_max(min_values, max_values)
    print("Absolute Max Values:", absolute_max_values)

    current_scaler = round(absolute_max_values['I_T1_I1'])
    voltage_scaler = round(absolute_max_values['V_T1_I1'])

    # Define hyperparameters and their ranges
    models = ['AE', 'DAE', 'VAE']
    var_models = ['all', 'all_holes', '0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20']
    hidden_sizes = [20, 30, 40]
    torch_seed = [5, 6, 7, 8, 9]
    window_size = 100
    step_size = 100
    max_epochs = 100
    batchsize = 64

    # Create all combinations of hyperparameters
    hparam_combinations = product(models, var_models, hidden_sizes, torch_seed)

    # Create the "models" directory at the same level as "data"
    # For example, if path is "simulation/Modular_Electronics/data/model_T1"
    # then the parent directory of "data" is "simulation/Modular_Electronics"
    models_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory at {models_dir}")

    for model, var_model, hidden_size, torch_seed in hparam_combinations:
        model_filename = os.path.join(models_dir, f"{index}_{model}_{var_model}_{hidden_size}_{torch_seed}.pth")
        # this loop goes through the experiments
        csv_logger = CSVLogger('logs', name=f'my_model_{var_model}_{torch_seed}')
        #exp_results = [var_model, None, None, None, None, None, None, None, None, None]
        if var_model in ['all', 'all_small', 'all_holes']:
            train_file = f'{path}/train/RLC_T{index}_{var_model}.csv'
        else:
            train_file = f'{path}/train/RLC_T{index}_{var_model}_OK.csv'
        train_set = load_data(train_file, window_size, step_size, voltage_scaler, current_scaler)
        # Scale Data
        print(train_set.shape)
        
        L.seed_everything(torch_seed, workers=True)

        train_set_size = int(train_set.shape[0] * 0.8)
        valid_set_size = train_set.shape[0] - train_set_size
        train_set, valid_set = torch.split(train_set, [train_set_size, valid_set_size], dim=0)
        #train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

        train_set = TimeSeriesDataset(train_set)
        valid_set = TimeSeriesDataset(valid_set)

        train_loader = DataLoader(
            train_set,
            batch_size=batchsize,
            shuffle=True,
            num_workers=8
        )
        
        valid_loader = DataLoader(
            valid_set,
            batch_size=batchsize,
            shuffle=False,
            num_workers=8
        )

        # Create a Namespace object for hyperparameters
        hparams = Namespace(
            hidden_size=hidden_size,
        )

        custom_checkpoint_callback = ModelCheckpoint(
            #every_n_epochs=1,
            save_top_k=-1,  # Keep all checkpoints
            monitor='val_loss',
            dirpath=f'./custom_checkpoints/RLC/{index}/',
            filename= f'model_{var_model}' + '{epoch:02d}_{val_loss:.4f}'
            )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=1e-4,  # allows small but real improvements
            mode='min',
            verbose=False
        )

        trainer = L.Trainer(callbacks=[early_stopping_callback],
                            logger=csv_logger,
                            limit_train_batches=100,
                            max_epochs=max_epochs,
                            log_every_n_steps=1,
                            deterministic=True,
                            fast_dev_run=fast_dev_run)

        # init model and trainer
        match model:
            case 'AE':
                autoencoder = LitAutoEncoder(hparams)
        
            case 'DAE':
                autoencoder = LitDenoisingAutoEncoder(hparams)
        
            case 'VAE':
                autoencoder = LitVariationalAutoEncoder(hparams)
        
            case _:
                raise ValueError(f"Unknown model type: {model}")

        # Check if the model file exists
        if os.path.exists(model_filename) and not retrain:
            autoencoder.load_state_dict(torch.load(model_filename))
            print(f"Loaded existing model from {model_filename}")
        else:
            print(f"Training new model and saving to {model_filename}")
            trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
            torch.save(autoencoder.state_dict(), model_filename)
            print(f"Model saved to {model_filename}")

        if test and not fast_dev_run:
            test_cases = ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20']
            # model, var_model, hidden_size, torch_seed in hparam_combinations
            if not os.path.exists(f"./filelogs/T"):
                # Create the folder if it doesn't exist
                os.makedirs(f"./filelogs/T")
                print(f"Folder ./filelogs/T created.")
            else:
                pass

            # Model testing
            autoencoder.eval()
            with torch.no_grad():
                for test_case in test_cases:
                    logdir = f"./filelogs/T/{index}_{model}_{var_model}_{hidden_size}_{torch_seed}_{test_case}"

                    # Test case normal system
                    test_ok_file = f'{path}/test/RLC_T{index}_{test_case}_OK.csv'
                    test_ok_set = load_data(test_ok_file, window_size, step_size, voltage_scaler, current_scaler)
                    test_ok_set = TimeSeriesDataset(test_ok_set)
                    ok_test_log = []
                    for idx in range(len(test_ok_set)):
                        sample = test_ok_set[idx]
                        residual, original, reconstruction = autoencoder(sample)
                        ok_test_log.append(residual)
                    logdir_ok = logdir + "_OK.pth"
                    torch.save(ok_test_log, logdir_ok)
                    
                    # Test case capacitance anomaly
                    if C:
                        test_cap_file = f'{path}/anom/RLC_T{index}_{test_case}_C.csv'
                        test_cap_set = load_data(test_cap_file, window_size, step_size, voltage_scaler, current_scaler)
                        test_cap_set = TimeSeriesDataset(test_cap_set)
                        cap_test_log = []
                        for idx in range(len(test_cap_set)):
                            sample = test_cap_set[idx]
                            residual, original, reconstruction = autoencoder(sample)
                            cap_test_log.append(residual)
                        logdir_cap = logdir + "_C.pth"
                        torch.save(cap_test_log, logdir_cap)
                    
                    if I:
                        # Test case inductance anomaly
                        test_ind_file = f'{path}/anom/RLC_T{index}_{test_case}_I.csv'
                        test_ind_set = load_data(test_ind_file, window_size, step_size, voltage_scaler, current_scaler)
                        test_ind_set = TimeSeriesDataset(test_ind_set)
                        ind_test_log = []
                        for idx in range(len(test_ind_set)):
                            sample = test_ind_set[idx]
                            residual, original, reconstruction = autoencoder(sample)
                            ind_test_log.append(residual)
                        logdir_ind = logdir + "_I.pth"
                        torch.save(ind_test_log, logdir_ind)
                    
                    if R:
                        # Test case resistance anomaly
                        test_res_file = f'{path}/anom/RLC_T{index}_{test_case}_R.csv'
                        test_res_set = load_data(test_res_file, window_size, step_size, voltage_scaler, current_scaler)
                        test_res_set = TimeSeriesDataset(test_res_set)
                        res_test_log = []
                        for idx in range(len(test_res_set)):
                            sample = test_res_set[idx]
                            residual, original, reconstruction = autoencoder(sample)
                            res_test_log.append(residual)
                        logdir_res = logdir + "_R.pth"
                        torch.save(res_test_log, logdir_res)

# Set up logging to file
logging.basicConfig(
    filename='RLC_error_log.txt',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

if __name__ == "__main__":
    # <Settings_>
    fast_dev_run = False  # True: activate lightning dev run; False: Normal operation
    test = True
    retrain = True
    torch.set_float32_matmul_precision('high')
    # <_Settings>

    paths = ["simulation/Modular_Electronics/data/model_T1", "simulation/Modular_Electronics/data/model_T2", "simulation/Modular_Electronics/data/model_T3", "simulation/Modular_Electronics/data/model_T4", "simulation/Modular_Electronics/data/model_T5"]
    for path in paths:
        if not os.path.exists(path):
            print(f"Folder {path} not available")
        else:
            print(f"Running main with {path}")
            try:
                main(path)
            except KeyboardInterrupt:
                print("KeyboardInterrupt received. Exiting...")
                sys.exit(0)
            except Exception as e:
                logging.error(f"Error with path '{path}': {e}", exc_info=True)
                print(f"Error occurred for {path}, check error_log.txt")
                continue