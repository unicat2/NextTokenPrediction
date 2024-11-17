import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import *
from model.LSTM import LSTMModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_args_parser():

    parser = argparse.ArgumentParser('next-token-predition', add_help=False)

    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--grad_clip', type=float, default=3.0)
    # Model parameters
    parser.add_argument('--model', default='AttentionModel', type=str, metavar='MODEL')
    # sequence length
    parser.add_argument('--max_len', type=int, default=512)
    # vocab size
    parser.add_argument('--vocab_size', type=int, default=21128)
    # window size
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=128)

    # Dataset
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
    parser.add_argument('--output_dir', default='./output_rnn')
    parser.add_argument('--log_dir', default='./output_rnn')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='NGramDataset.py')
    parser.add_argument('--num_workers', default=10, type=int)



    return parser


def main(args):

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    model = LSTMModel(args.vocab_size, args.embedding_dim, args.hidden_dim).to(args.device)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # prepare dataset
    train_dataset = DatasetSentence(args.train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset) if args.distributed else None
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                  shuffle=(train_sampler is None), num_workers=4)

    test_dataset = DatasetSentence(args.test_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset) if args.distributed else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                                 shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler()

    log_writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir and (
            not args.distributed or dist.get_rank() == 0) else None

    train_log_file = open(os.path.join(args.output_dir, 'train_log.txt'), 'w')
    eval_log_file = open(os.path.join(args.output_dir, 'eval_log.txt'), 'w')

    def adjust_learning_rate(epoch):
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(epoch):
        if args.output_dir and (not args.distributed or dist.get_rank() == 0) and (
                epoch % 4 == 0 or epoch + 1 == args.epochs):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

    def train_one_epoch(epoch):
        model.train()
        total_loss = 0.0
        if train_sampler:
            train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):

            tokens = tokenizer(batch, truncation=True, padding=True, max_length=args.max_len, return_tensors='pt').to(args.device)
            input_ids = tokens['input_ids'][:, :-1]
            label_ids = tokens['input_ids'][:, 1:].clone()
            label_ids[label_ids == tokenizer.pad_token_id] = -100

            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                loss = criterion(logits.view(-1, args.vocab_size), label_ids.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

            total_loss += loss.item()
            if log_writer:
                log_writer.add_scalar('train_loss', loss.item(), epoch * len(train_dataloader) + step)

        avg_loss = total_loss / len(train_dataloader)
        train_log_file.write(f'Epoch {epoch}, Train Loss: {avg_loss}\n')
        print(f'Epoch {epoch}, Train Loss: {avg_loss}')

    def evaluation(epoch):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                tokens = tokenizer(batch, truncation=True, padding=True, max_length=args.max_len,
                                   return_tensors='pt').to(args.device)
                input_ids = tokens['input_ids'][:, :-1]
                label_ids = tokens['input_ids'][:, 1:].clone()
                label_ids[label_ids == tokenizer.pad_token_id] = -100

                with torch.cuda.amp.autocast():
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, args.vocab_size), label_ids.view(-1))

                total_loss += loss.item()

        avg_loss = total_loss / len(test_dataloader)
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








