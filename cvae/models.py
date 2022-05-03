import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, num_labels=0):

        super().__init__()


        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, num_labels)

    def forward(self, x, c1=None,c2=None,c3=None,c4=None):

        if x.dim() > 2:
            x = x.view(-1, 64*64)

        means, log_var = self.encoder(x, c1,c2,c3,c4)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c1,c2,c3,c4)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c1=None,c2=None,c3=None,c4=None):

        recon_x = self.decoder(z, c1,c2,c3,c4)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size,  num_labels):

        super().__init__()


        layer_sizes[0] += 4

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            print("in_size",in_size)
            print("out_size",out_size)
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c1=None,c2=None,c3=None,c4=None):


        c1=(c1/64.).reshape(-1,1)
        c2=(c2/64.).reshape(-1,1)
        c3=(c1/64.).reshape(-1,1)
        c4=(c2/64.).reshape(-1,1)
        x = torch.cat((x, c1,c2,c3,c4), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size,  num_labels):

        super().__init__()

        self.MLP = nn.Sequential()
        input_size = latent_size + 4


        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c1,c2,c3,c4):

            
        c1=(c1/64.).reshape(-1,1)
        c2=(c2/64.).reshape(-1,1)
        c3=(c1/64.).reshape(-1,1)
        c4=(c2/64.).reshape(-1,1)
            

        z = torch.cat((z, c1,c2,c3,c4), dim=-1)


        x = self.MLP(z)

        return x
