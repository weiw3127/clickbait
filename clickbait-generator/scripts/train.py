import argparse
import logging
from clickbait.dataset import ClickbaitDataset
from clickbait.train import train_model, TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BART on DailyMail clickbait data.")
    parser.add_argument("--csv", default="data/raw/dailymail.csv", help="Input CSV with article,headline.")
    parser.add_argument("--output", default="models/bart-clickbait", help="Model output directory.")
    parser.add_argument("--model", default="facebook/bart-base", help="Base model name.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 (if GPU).")
    parser.add_argument("--val-size", type=float, default=0.1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Build dataset
    ds_builder = ClickbaitDataset(tokenizer_name=args.model)
    dset = ds_builder.build_from_csv(args.csv, val_size=args.val_size)

    # Train
    cfg = TrainConfig(
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        train_bs=args.train_batch,
        eval_bs=args.eval_batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
    )
    train_model(dset, cfg, tokenizer=ds_builder.tokenizer)


if __name__ == "__main__":
    main()
