import argparse
from clickbait.generate import TitleGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate a clickbait headline from an article.")
    parser.add_argument("--model-dir", default="models/bart-clickbait", help="Fine-tuned model directory.")
    parser.add_argument("--article", type=str, help="Raw article text.")
    parser.add_argument("--file", type=str, help="Path to a .txt file containing the article.")
    parser.add_argument("--max-length", type=int, default=24)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--sample", action="store_true", help="Use sampling instead of beam search.")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    if not args.article and not args.file:
        raise SystemExit("Provide --article or --file")

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            article = f.read()
    else:
        article = args.article

    tg = TitleGenerator(model_dir=args.model_dir)
    title = tg.generate(
        article=article,
        max_length=args.max_length,
        num_beams=args.num_beams,
        do_sample=args.sample,
        temperature=args.temperature,
    )
    print(title)


if __name__ == "__main__":
    main()
