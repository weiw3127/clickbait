# Clickbait Title Generator

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Model-bwlw3127/clickbait--bart--dailymail-blue)](https://huggingface.co/bwlw3127/clickbait-bart-dailymail)

Fine-tuned [`facebook/bart-base`](https://huggingface.co/facebook/bart-base) on DailyMail article/headline pairs to generate **clickbait-style headlines**.  
👉 Model is hosted on Hugging Face Hub: [bwlw3127/clickbait-bart-dailymail](https://huggingface.co/bwlw3127/clickbait-bart-dailymail)



---
In the media industry, we were often held hostage by a classic and relentless KPI: click-through rate (CTR). Yet, a great article meant nothing if no one clicked. To draw people's attention, we leaned on headlines that are sensational, intriguing, sometimes exaggerated, but never deceptive, since credibility is essential to our reputation. 

Clickbait is an art, balancing creativity, tone, and honesty. It’s a linguistic trick, refined by human intuition. But somehow I began to wonder: could a model do it too?

That led to this experiment. I fine-tuned Facebook’s BART based on DailyMail headlines and articles, a publication notorious for its clickbait mastery. The goal: build a model that can generate irresistible headlines and test whether AI can rival human copywriters.

To train a clickbait-style headline generator, I built a dataset from DailyMail, known for its sensational tone. I focused on Showbiz, News, TV & Royals, which rich in provocative, high-engagement headlines. From each article, I paired the article content (input) with its headline (target), yielding 3,354 article–headline pairs. These serve as supervised training examples: given the article, generate its headline.

## Clickbait Generating Example

Input article:

> Country Road is fighting for its future largely thanks to a cost-cutting decision the company made more than 20 years ago, an expert says. The once-beloved Aussie brand is in clear trouble, with Country Road Group reporting in March its earnings were down almost 72 per cent at $14.2million for the last half of 2024.

> One of its longstanding flagship stores at Sydney CBD's Queen Victoria Building has shut up shop, as has sister brand Trenery in Mosman, on Sydney's affluent lower north shore. Another CBD store in Sydney's Pitt Street Mall is expected to close when its lease expires in three years' time.

> The video player is currently playing an ad. You can skip the ad in 5 sec with a mouse or keyboard
Queensland University of Technology marketing expert Gary Mortimer said Country Road had lost its iconic Australian lifestyle brand status since Woolworths Holdings took a controlling stake in the late 90s.

> A cost-cutting move to manufacture offshore gradually eroded its 'Made in Australia' appeal and weakened the brand's authenticity, Professor Mortimer said.

> 'Since its launch in the mid-1970s, Country Road clothing was primarily made in Australia, specifically, the iconic chambray shirt which I and nearly every other young man had during that time was made here using Australian cotton,' he said.

> 'The company emphasised its commitment to Australian manufacturing during that time. 

> 'Much of that production has shifted to Bangladesh, China, India and Pakistan under new ownership, essentially losing the essence of what Country Road stood for.'

Generated headline:

> Country Road is in clear trouble thanks to cost-cutting decision the company made 20 years ago


## Project Structure
```bash
clickbait-generator/
├─ clickbait/                 # core package
│  ├─ scraper.py              # scrape DailyMail articles
│  ├─ dataset.py              # dataset building + tokenization
│  ├─ train.py                # fine-tuning logic
│  └─ generate.py             # inference wrapper
├─ scripts/                   # CLI entry points
│  ├─ scrape.py
│  ├─ train.py
│  └─ generate.py
├─ data/                      # optional local storage (raw data, etc.)
├─ requirements.txt
└─ README.md

```

## Try it yourself
Generate a headline directly from the Hub:
```bash
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "bwlw3127/clickbait-bart-dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

article = "Long article text..."
inputs = tokenizer("generate clickbait headline: " + article,
                   return_tensors="pt", truncation=True, max_length=512)

outputs = model.generate(**inputs, max_length=24, num_beams=4, early_stopping=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Or from CLI (uses the Hugging Face model by default):
```bash
python scripts/generate.py --article "Long article text..."
```

## Notes

+ Data Source: scraped DailyMail articles (for demo/educational use).

+ Bias & Style: reflects sensationalism typical of the source.

+ Use Cases: not for factual reporting, but good as a demo of fine-tuning seq2seq models.

## Citation

If you use this model, please cite the repository:

```bibtex
@software {wei_clickbait_generator_2025,
  author       = {Wei-Ling, W.},
  title        = {Clickbait Title Generator},
  month        = aug,
  year         = 2025,
  publisher    = {GitHub},
  url          = {https://github.com/weiw3127/clickbait},
  note         = {Fine-tuned BART model for clickbait headline generation.}
}
```

**Model**
```bibtex
@misc {clickbaitbart2025,
  author = {Wei-Ling, W.},
  title = {Clickbait BART (DailyMail Fine-tune)},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/bwlw3127/clickbait-bart-dailymail}}
}
```

