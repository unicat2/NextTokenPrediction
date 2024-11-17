import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import *
from model.FFN import FeedForwardNetwork

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_args_parser():
    parser = argparse.ArgumentParser('next-token-predition', add_help=False)

    # Training parameters
    parser.add_argument('--batch_size', default=4096, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--grad_clip', type=float, default=3.0)
    # Model parameters
    parser.add_argument('--model', default='AttentionModel', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=21128)
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=64)
    # parser.add_argument('--head', type=int, default=4)
    # Dataset parameters
    parser.add_argument('--train_data', type=str, default='../Data/train_data_2017.txt')
    parser.add_argument('--test_data', type=str, default='../Data/val_data_2017.txt')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='NGramDataset.py',
                        help='epochs to warmup LR')
    # Training parameters
    parser.add_argument('--output_dir', default='./output_ffn_2',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_ffn_2',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='NGramDataset.py',help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)



    return parser



def main(args):

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    model = FeedForwardNetwork(args.vocab_size, args.embedding_dim, args.window_size, args.hidden_dim).to(args.device)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    train_log_file = open(os.path.join(args.output_dir, 'train_log.txt'), 'w')
    eval_log_file = open(os.path.join(args.output_dir, 'eval_log.txt'), 'w')

    # prepare dataset
    train_dataset = DatasetTokenized(args.train_data, args.window_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset) if args.distributed else None
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                  shuffle=(train_sampler is None), num_workers=4)

    test_dataset = DatasetTokenized(args.test_data, args.window_size)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset) if args.distributed else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                                 shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler()

    log_writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir and (
            not args.distributed or dist.get_rank() == 0) else None

    train_batch_count = None
    test_batch_count = None

    def _compute_batch_count(dataloader):
        if hasattr(dataloader.dataset, '__len__'):
            return len(dataloader)
        else:
            count = 0
            for _ in dataloader:
                count += 1
            return count

    def adjust_learning_rate(epoch):
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(epoch):
        if args.output_dir and (not args.distributed or dist.get_rank() == 0) and (epoch % 4 == 0 or epoch + 1 == args.epochs):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

    def train_one_epoch(epoch):
        nonlocal train_batch_count
        if train_batch_count is None:
            train_batch_count = _compute_batch_count(train_dataloader)

        model.train()
        total_loss = 0.0
        if train_sampler:
            train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            input_ids = batch[0].to(args.device)
            label_ids = batch[1].to(args.device)

            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                loss = criterion(logits, label_ids)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

            total_loss += loss.item()
            if log_writer:
                log_writer.add_scalar('train_loss', loss.item(), epoch * train_batch_count + step)

        avg_loss = total_loss / train_batch_count
        train_log_file.write(f'Epoch {epoch}, Train Loss: {avg_loss}\n')
        print(f'Epoch {epoch}, Train Loss: {avg_loss}')

    def evaluation(epoch):
        nonlocal test_batch_count
        if test_batch_count is None:
            test_batch_count = _compute_batch_count(test_dataloader)

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                input_ids = batch[0].to(args.device)
                label_ids = batch[1].to(args.device)

                with torch.cuda.amp.autocast():
                    logits = model(input_ids)
                    loss = criterion(logits, label_ids)

                total_loss += loss.item()

        avg_loss = total_loss / test_batch_count
        eval_log_file.write(f'Epoch {epoch}, Test Loss: {avg_loss}\n')
        print(f'Epoch {epoch}, Test Loss: {avg_loss}')
        if log_writer:
            log_writer.add_scalar('test_loss', avg_loss, epoch)

    for epoch in range(args.epochs):
        adjust_learning_rate(epoch)
        train_one_epoch(epoch)
        evaluation(epoch)
        save_checkpoint(epoch)




if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    device = torch.device(f'cuda:{args.local_rank}' if args.distributed else args.device)

    main(args)


