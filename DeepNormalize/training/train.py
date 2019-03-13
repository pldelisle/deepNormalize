import torch

from DeepNormalize.layers.tversky_loss import TverskyLoss


class Trainer(object):
    def __init__(self, generator, discriminator, data_provider, config_generator, config_discriminator):
        self._generator = generator
        self._discriminator = discriminator
        self._data_provider = data_provider
        self._config_G = config_generator
        self._config_D = config_discriminator

    def train(self):
        training_dataloader, validation_dataloader = self._data_provider.get_data_loader()

        optimizer_G = torch.optim.Adam(self._generator.parameters(), self._config_G.get("learning_rate"))

        learning_rate_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G,
                                                                         milestones=[30, 60],
                                                                         gamma=0.1)

        optimizer_D = torch.optim.Adam(self._discriminator.parameters(), self._config_D.get("learning_rate"))

        learning_rate_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D,
                                                                         milestones=[30, 60],
                                                                         gamma=0.1)

        segmentation_loss = TverskyLoss(alpha=self._config_G.get("alpha"),
                                        beta=self._config_G.get("beta"),
                                        eps=self._config_G.get("eps"),
                                        n_classes=self._config_G.get("n_classes"))

        adversarial_loss = torch.nn.CrossEntropyLoss()

        real_label = 1
        fake_label = 0

        for epoch in range(100):

            learning_rate_scheduler_G.step()
            learning_rate_scheduler_D.step()

            for i, (inputs, labels) in enumerate(training_dataloader):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with real.

                optimizer_D.zero_grad()

                real = inputs.float().cuda()
                batch_size = real.size(0)
                label = torch.full((batch_size, ), real_label, device="cuda")

                output = self._discriminator(real)
                loss_D_real = adversarial_loss(output, label.long())
                loss_D_real.backward(retain_graph=True)

                # Train with generated samples.
                fake_normalized, _ = self._generator(inputs.float().cuda())
                label.fill_(fake_label)
                output = self._discriminator(fake_normalized)
                loss_D_fake = adversarial_loss(output, label.long())
                loss_D_fake.backward(retain_graph=True)

                error_D = loss_D_real + loss_D_fake

                optimizer_D.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                optimizer_G.zero_grad()

                _, output = self._generator(inputs.float().cuda())
                loss_G = segmentation_loss(output, labels.float().cuda())

                total_loss = error_D + loss_G

                total_loss.backward()
                optimizer_G.step()
                print(total_loss.cpu().detach().numpy())
