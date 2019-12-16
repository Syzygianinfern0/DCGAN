import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from config import *
from loader import load_imgs
from models import Generator, Discriminator
from weight_init import weights_init

writer = SummaryWriter()


def main():
    #################################
    # ~~~~~~ MODELS AND DATA ~~~~~~ #
    #################################

    net_g: nn.Module = Generator()
    net_g.to(device)
    if device.type == 'cuda':
        net_g = nn.DataParallel(net_g, [0])
    net_g.apply(weights_init)
    # print(net_g)

    net_d: nn.Module = Discriminator()
    net_d.to(device)
    if device.type == 'cuda':
        net_d = nn.DataParallel(net_d, [0])
    net_d.apply(weights_init)
    # print(net_d)

    # dataloader = load_imgs(subset=5000)
    dataloader = load_imgs()

    #######################################
    # ~~~~~~ OPTIMIZERS AND LOSSES ~~~~~~ #
    #######################################

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label = 1
    fake_label = 0

    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))

    ###############################
    # ~~~~~~ TRAINING LOOP ~~~~~~ #
    ###############################

    # Lists to keep track of progress
    img_list = []
    g_losses = []
    d_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            #########################################
            # Update D network:                     #
            # maximize log(D(x)) + log(1 - D(G(z))) #
            #########################################
            # Train with all-real batch
            net_d.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)

            # Forward pass real batch through D
            output = net_d(real_cpu).view(-1)

            # Calculate loss on all-real batch
            err_d_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            err_d_real.backward()
            d_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = net_g(noise)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = net_d(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            err_d_fake = criterion(output, label)

            # Calculate the gradients for this batch
            err_d_fake.backward()
            d_g_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            err_d = err_d_real + err_d_fake

            # Update D
            optimizer_d.step()

            #########################
            # Update G network:     #
            # maximize log(D(G(z))) #
            #########################
            net_g.zero_grad()

            # Fake labels are real for generator cost
            label.fill_(real_label)

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = net_d(fake).view(-1)
            # Calculate G's loss based on this output
            err_g = criterion(output, label)

            # Calculate gradients for G
            err_g.backward()
            d_g_z2 = output.mean().item()

            # Update G
            optimizer_g.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         err_d.item(), err_g.item(), d_x, d_g_z1, d_g_z2))
                writer.add_scalar('val/d_x', d_x, iters)
                writer.add_scalar('val/d_g_z1', d_g_z1, iters)
                writer.add_scalar('val/d_g_z2', d_g_z2, iters)

            # Save Losses for plotting later
            g_losses.append(err_g.item())
            d_losses.append(err_d.item())

            writer.add_scalar('loss/g', err_g.item(), iters)
            writer.add_scalar('loss/d', err_d.item(), iters)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = net_g(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                writer.add_images('generated', fake, iters)
            iters += 1


if __name__ == '__main__':
    main()
    writer.close()
