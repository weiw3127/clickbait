import { GoogleGenAI } from "@google/genai";
import dotenv from 'dotenv';

dotenv.config();

class ClickbaitGenerator{
    constructor(){
        this.ai = new GoogleGenAI({
            apiKey: process.env.GEMINI_API_KEY
        });
        this.model = "gemini-2.5-flash";
    }

    generatePrompt(article, options={}){
        const {
            style = 'sensational',
            count = 3,
            targetAudience = 'general',
            includeNumbers = true,
            includeEmojis = false
        } = options;

        let prompt = `
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

        Generate exactly ${count} different clickbait titles, numbered 1-${count}:`;

        return prompt;
    }

    async generateTitles(article, options={}){
        try{
            if(!article || article.trim().length === 0){
                throw new Error('Article conctent is required');
            }

            const prompt = this.generatePrompt(article, options);
            console.log('generating clickbait titles ... \n');
            const response = await this.ai.models.generateContent({
                model: this.model,
                contents: prompt,
            })
            const titles = response.text; 
            console.log('✅ Generated Titles:');
            console.log('==================');
            console.log(titles);
            console.log('==================\n');
            
            return {
                success: true,
                titles: titles,
                options: options,
                articleLength: article.length
            };

        } catch (error){
            console.error('Error Generating titles: ', error.message);
            return{
                sucess: false,
                error: error.message
            };
        }
    }

}

async function main(){
    const generator = new ClickbaitGenerator();
    const article = `He needed two surgeries afterwards

    An adrenaline junkie fractured his skull and needed two major surgeries after he attempted a world-record 'death dive' from an Australian waterfall.

    Vali Graham, 21, suffered horrific injuries after he leapt from the top of Minnehaha Falls, a 42.5metre cliff, in the NSW Blue Mountains on June 11. He has undergone multiple operations and weeks of rehabilitation after the dive.

    Footage of the death-defying stunt showed Mr Graham gearing himself up at the top of the waterfall on a rock ledge before he jumped off and performed an acrobatic twist in the air on the way down.

    Unfortunately he landed in an awkward 'pike' position when he hit the water.

    The impact knocked him out, fractured his skull and back, caused a concussion and burst his eardrum.

    He had a safety team of several 'spotters' at the bottom of the waterfall who jumped in the pond to assist him.

    Incredibly he came to in the water and managed to swim, assisted, to the edge and pull himself onto solid ground despite his extensive injuries.

    The Newcastle, NSW native was rushed to hospital where he underwent emergency surgeries.

    The stuntman provided an update to his followers a couple of days later letting everybody know he was, eventually, going to be fine.

    Death diving, also called dødsing, is an extreme sport that was pioneered in Norway.

    The current world record is held by Swiss diver Lucien Charlon with a height of 41.7 meters.

    In the amateur sport divers jump from high platforms and outstretch themselves in the air then curl into a pike, or ball, position before hitting the water.

    Vali Graham, 21, fractured several bones, burst his eardrum and suffered a concussion after jumping off the top of Minnehaha Falls, a 42.5 metre waterfall, in the Blue Mountains on June 11

    The death-diving enthusiast said his suffering was a gift of god in an update posted after the incident

    Minnehaha Falls in the Blue Mountains features a 42.5metre sheer cliff

    There are competitions held for the sport in some countries, along with a tour and the world championships, which are held in Norway.

    The drop from the rock ledge at Minnehaha Falls to the water was about as high as a 13-storey building.

    His suffering, Mr Graham wrote, was a gift from God.

    'Update: 'God gives us the gift of suffering' after sending (sic) this monster 42.5m cliff I was knocked unconscious,' Mr Graham wrote.

    'I sustained a burst eardrum and fractured my T11 vertebrae, sternum, and a small fracture on my skull near my burst eardrum. All love to my spotters.

    'After regaining consciousness I pulled myself out of the water and walked a steep 1.2km out to our car where my friends took me to hospital.'

    Mr Graham also thanked his followers for the love and support.

    'The support was amazing, I've had surgery on my back and sternum and was walking 2 days after surgery,' he wrote.

    'I am honestly mentally feeling amazing, ready to rebuild my body better than ever and come back stronger, a long way to go but excited for the journey.'

    His followers were floored by his dedication to the sport.

    'Beyond comprehension how he does that,' one person commented.

    'Full credit to him he deserves the world record,' another wrote.

    However, some who watched the video had less-than-stellar feedback for Mr Graham.

    'My taxes have to pay for your dumb choices,' one person commented.

    'God didn't give you that suffering, you did that yourself,' another said.`
    
    console.log('=== BASIC CLICKBAIT GENERATION ===');
    const basicResult = await generator.generateTitles(article);
}

main().catch(console.error);

export { ClickbaitGenerator };