import { Anthropic } from '@anthropic-ai/sdk';
import dotenv from 'dotenv';
dotenv.config();

class ClickbaitGenerator {
    constructor() {
        this.anthropic = new Anthropic({
            apiKey: process.env.ANTHROPIC_API_KEY
        });
        this.model = 'claude-3-opus-20240229'; // or claude-3-sonnet for cheaper, claude-3-haiku for fastest
    }

    generatePrompt(article, options = {}) {
        const {
            style = 'sensational',
            count = 3,
            targetAudience = 'general',
            includeNumbers = true,
            includeEmojis = false
        } = options;

        return `
As an expert clickbait title writer, create ${count} ${style} titles for this article.

REQUIREMENTS:
- Make titles attention-grabbing and curiosity-inducing
- Use power words and emotional triggers
- Keep titles under 60 characters for social media optimization
- Target audience: ${targetAudience}
${includeNumbers ? '- Include numbers when relevant (e.g., "5 Ways...", "This 21-Year-Old...")' : ''}
${includeEmojis ? '- Add relevant emojis sparingly' : '- Do not use emojis'}

CLICKBAIT TECHNIQUES TO USE:
- Shock factor and surprise
- Create curiosity gaps ("You Won't Believe What Happened Next...")
- Use superlatives (amazing, incredible, shocking)
- Personal stories angle
- Consequence-focused headlines

ARTICLE:
"${article}"

Generate exactly ${count} different clickbait titles, numbered 1-${count}.
        `;
    }

    async generateTitles(article, options = {}) {
        try {
            if (!article || article.trim().length === 0) {
                throw new Error('Article content is required');
            }

            const prompt = this.generatePrompt(article, options);
            console.log('generating clickbait titles with Claude ... \n');

            const response = await this.anthropic.messages.create({
                model: this.model,
                max_tokens: 1000,
                messages: [
                    {
                        role: "user",
                        content: prompt
                    }
                ]
            });

            const titles = response.content?.[0]?.text || '[No titles returned]';
            console.log('✅ Generated Titles:\n' + titles + '\n');
            return {
                success: true,
                titles,
                options,
                articleLength: article.length
            };

        } catch (error) {
            console.error('Error Generating titles: ', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }
}

export { ClickbaitGenerator };
