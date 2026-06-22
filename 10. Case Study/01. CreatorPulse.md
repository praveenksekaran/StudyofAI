# Problem statement - Creator Pulse
CreatorPulse is a daily co-pilot for Twitter/X, Linkedin, and Instagram creators.
It creates ready-to-post drafts, bundles trend insights and a feedback loop that keeps learning the creators voice. The goal is to cut ideation time from 1-2 hours to 15< min per post while lifting impressions 2-3x.

- Source: Twitter, Youtube, RSS
- Reasearch & trend Engine: Nightly crawl --> Vector Summirization --> spike detection 
- Writing style trainer 
- Draft generator 
  - Twitter: single tweets, 8-12 tweet threads, quote tweet replies 
  - Linkedin: short (<300 words) posts long articles, carousal outlines
  - Insta: reel scripts 
- Morning Delivery : at 8 via Email or whatsapp, includes, ready to post drafts and trends to watch 
- Feedback loop : emoji in channel, auto diff on edits, feeds style and ranking 
- Responsive web dashboard: post performance analytics 

# Solution

#### Build a PRD 
- Use the prompt to generate PRD 
[prompt](https://github.com/Siddhant-Goswami/100x-LLM/blob/main/prompts/PRODUCT_MANAGER.md)

- Twitter --> Apify.com --> Data pre processing (LLM) --> Get the trending topics --> Deep research (Augumented LLM) --> Report/Summary --> send to email 8 AM --> Human review 
- (Style & format) writter LLM (send the Report/Summary) --> First draft --> Human review --> Post on twitter/Linkedin --> Post fetch analytics (likes, impressions, comments, shares) 
- Augumented LLM (Post analytics) --> writter LLM (for changes)

#### Lovable to build UI
- Use lovable to connect with Supabase for data
- check the code to Git

#### Calude code for back end 
- connect Claude code with Git
- Prompt Claude code for API specs for every operation (auto identifies with code and supabase)
- prompt Claude code to build the API connected with frontend
- 

#### Sources
web scraper (twitter/Insta): Apify.com or firecrawl.dev or selinium web crawler

