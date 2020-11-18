# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

class eegdg_logger(SummaryWriter):
    def __init__(self, logdir):
        super(eegdg_logger, self).__init__(logdir)

    def log_training(self, train_loss, train_acc, val_loss, val_acc, test_loss,test_acc, lr, epoch):
            self.add_scalar("train/loss", train_loss, epoch)
            self.add_scalar("validation/loss", val_loss, epoch)
            self.add_scalar("test/loss", test_loss, epoch)
            self.add_scalar("train/acc", train_acc, epoch)
            self.add_scalar("validation/acc", val_acc, epoch)
            self.add_scalar("test/acc", test_acc, epoch)
            self.add_scalar("train/lr",lr,epoch)