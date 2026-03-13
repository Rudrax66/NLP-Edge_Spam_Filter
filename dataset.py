"""
Synthetic Dataset Generator
Produces realistic labelled samples for spam, ham, and toxic text.
Labels:  0 = ham (clean)  |  1 = spam  |  2 = toxic
"""

import random
from typing import List, Tuple

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Ham samples (clean, normal messages)
# ─────────────────────────────────────────────────────────────────────────────
HAM_SAMPLES = [
    "Hey, are you free to grab coffee tomorrow morning?",
    "Please find the attached report for your review.",
    "The meeting has been rescheduled to 3 PM on Friday.",
    "Can you pick up some milk on your way home?",
    "I really enjoyed the book you recommended last week.",
    "The project deadline has been extended by two days.",
    "Happy birthday! Hope you have a wonderful day.",
    "Just checking in to see how you're doing.",
    "The weather forecast says it will rain tomorrow.",
    "Thanks for your help with the presentation yesterday.",
    "I'll be working from home on Monday.",
    "The new software update fixed the login issue.",
    "Dinner is ready when you get home.",
    "Please review the attached contract before signing.",
    "The quarterly results exceeded our expectations this year.",
    "Let me know if you need any help with the assignment.",
    "The flight has been delayed by 30 minutes.",
    "Could you send me the file we discussed earlier?",
    "I'll be in the office by 9 AM tomorrow.",
    "Great job on the presentation today, everyone!",
    "The library closes at 8 PM on weekdays.",
    "Your package has been shipped and will arrive Thursday.",
    "We've scheduled the team retrospective for next week.",
    "The new café around the corner has great pastries.",
    "Please remember to submit your timesheet by Friday.",
    "The conference call starts in 10 minutes.",
    "I left my umbrella in the conference room.",
    "The annual performance reviews begin next month.",
    "Could we reschedule our appointment to the afternoon?",
    "The kids' school play is on Saturday at 6 PM.",
    "I'm reading a really interesting book on machine learning.",
    "The doctor said everything looks good in my checkup.",
    "We need to order more printer paper for the office.",
    "The project documentation is now available on the wiki.",
    "Mom called — she wants to know if you're coming for dinner.",
    "The gym is offering a free trial membership this week.",
    "I've uploaded the latest design files to the shared folder.",
    "Our internet was down this morning but it's back now.",
    "The vegetarian option at today's lunch was surprisingly good.",
    "Please add your vacation days to the shared calendar.",
    "The report contains three main sections: intro, analysis, conclusion.",
    "We are happy to announce that our product launch was successful.",
    "Can you review the pull request I submitted this morning?",
    "The training session starts at 10 AM in the main hall.",
    "I've been learning Spanish for about three months now.",
    "The new employee handbook has been sent to everyone.",
    "Your subscription has been renewed successfully.",
    "We're planning a hiking trip next weekend.",
    "Please let us know your dietary preferences before the event.",
    "The budget has been approved for the next quarter.",
]

# ─────────────────────────────────────────────────────────────────────────────
# Spam samples
# ─────────────────────────────────────────────────────────────────────────────
SPAM_SAMPLES = [
    "CONGRATULATIONS! You've WON a FREE iPhone 15! CLICK HERE to claim NOW!!!",
    "You have been selected as the winner of our $1,000,000 lottery! Verify now.",
    "FREE OFFER: Buy 1 get 10 FREE! Limited time only. Act NOW before it expires!!!",
    "Urgent: Your account has been compromised. Click here to verify your password.",
    "Make $500 a day working from home! No experience needed. Limited spots left!",
    "EARN EXTRA INCOME from home! Guaranteed $5,000 per week. Click to learn more!",
    "Your PayPal account has been suspended. Confirm your details at http://paypa1.verify.com",
    "Hot singles in your area! Click here to meet them tonight. FREE registration!",
    "You've been chosen for a special prize! Call 1-800-555-0199 to claim your reward.",
    "Buy Cheap Meds Online! No prescription needed! 90% off retail prices! ORDER NOW!",
    "INVESTMENT OPPORTUNITY: 500% returns guaranteed! Send $100 to receive $500 back!",
    "FINAL NOTICE: Your computer has a virus! Download our FREE scanner immediately!",
    "Dear Friend, I am Prince Johnson and I need your help transferring $15,000,000 USD.",
    "Lose 30 lbs in 30 days! Our miracle pill GUARANTEED to work or money back!!!",
    "Your Netflix subscription expires TODAY. Update payment at http://netf1ix-bill.com",
    "URGENT: IRS final notice. You owe back taxes. Call 1-888-555-0123 IMMEDIATELY!",
    "Win a free vacation to Cancun! Just answer 3 questions! No purchase necessary!",
    "EXCLUSIVE OFFER for YOU only! Get rich quick with our proven investment system!",
    "Refinance your mortgage NOW! Lowest rates ever! Limited time offer. APPLY TODAY!",
    "FREE casino chips! Sign up now and get $500 bonus — no deposit required!!!",
    "DEAR LUCKY WINNER! Your email was selected. Claim your $750,000 prize NOW!",
    "We have a business proposal for you. Reply to claim your share of $45 million.",
    "Your Amazon order has an issue. Verify your account: http://amaz0n-secure.xyz",
    "MIRACLE CURE discovered by doctors! Big pharma doesn't want you to know this!",
    "Work online, earn thousands per week! Join 10,000 happy members TODAY for FREE!",
    "ALERT! Your bank account will be CLOSED unless you verify at http://bank-secure.info",
    "FREE iPhone! Complete a survey and get yours shipped FREE! Limited stock!!!",
    "Herbal supplements — 100% natural, 100% GUARANTEED! Buy 2 get 1 FREE NOW!",
    "Unclaimed inheritance funds in your name: $3.2 million. Contact agent John for details.",
    "SPECIAL: Click NOW to unlock your premium account for FREE! Offer expires midnight!",
    "Credit card debt relief — pay nothing! We'll erase your debt GUARANTEED! Call now.",
    "WARNING: Your computer is infected with 5 viruses! Download our cleaner IMMEDIATELY!",
    "You've won a $250 Best Buy gift card! Click here: http://bestbuy-rewards.net/claim",
    "Make money fast with crypto! 1000% profit in 30 days! Only 10 spots available!",
    "SEXY SINGLES want to chat! Join FREE today — over 1 million members online now!",
]

# ─────────────────────────────────────────────────────────────────────────────
# Toxic samples
# ─────────────────────────────────────────────────────────────────────────────
TOXIC_SAMPLES = [
    "You are the most stupid and worthless person I have ever met.",
    "I hate you so much. You should just disappear from everyone's life.",
    "You're an absolute idiot. How can someone be this dumb?",
    "Nobody likes you. Everyone thinks you're a loser and a failure.",
    "You should be ashamed of yourself. You're disgusting and horrible.",
    "I can't believe how pathetic and useless you are at everything.",
    "You're nothing but a liar and a manipulative piece of trash.",
    "Everyone hates you and no one will ever care about you.",
    "Stop talking. Your opinions are stupid and nobody wants to hear them.",
    "You're a waste of space and the world would be better without you.",
    "I despise you with every fiber of my being. Get out of my sight.",
    "You are the worst person alive. Completely disgusting and vile.",
    "Your existence is a mistake. Nobody could ever love something like you.",
    "You're so ugly and repulsive, it's honestly hard to look at you.",
    "I will make your life miserable. You deserve to suffer for what you did.",
    "You call yourself a professional? You're an embarrassment to the field.",
    "Shut up, you brain-dead moron. Your ideas are garbage and always will be.",
    "You're a bigot and a coward who hides behind a keyboard spreading hate.",
    "Go away. No one wants you here. You contribute absolutely nothing.",
    "You are literally the dumbest person I've ever had to deal with.",
    "I've never met someone as vile and disgusting as you in my entire life.",
    "You're a toxic, hateful, manipulative person who hurts everyone around you.",
    "You're pathetic. A complete failure who can't do anything right.",
    "Your stupidity is only matched by your arrogance. Truly insufferable.",
    "You are a racist, hateful person and you make everyone around you miserable.",
]


def generate_dataset(
    n_ham: int = 200,
    n_spam: int = 200,
    n_toxic: int = 200,
    augment: bool = True
) -> Tuple[List[str], List[int]]:
    """
    Returns (texts, labels) with:
      0 = ham, 1 = spam, 2 = toxic
    """
    texts, labels = [], []

    def augment_text(text: str) -> str:
        ops = [
            lambda t: t.upper() if random.random() < 0.15 else t,
            lambda t: t + " " + random.choice(["!", "!!", "!!!"]),
            lambda t: re.sub(r'\bfree\b', 'FREE', t, flags=re.I),
            lambda t: t.replace(".", ""),
            lambda t: " ".join(
                w.upper() if random.random() < 0.2 else w for w in t.split()),
        ]
        for op in ops:
            if random.random() < 0.3:
                text = op(text)
        return text

    import re

    # Ham
    for _ in range(n_ham):
        t = random.choice(HAM_SAMPLES)
        if augment and random.random() < 0.2:
            t = t + " " + random.choice(["Thanks!", "Best regards.", "See you then."])
        texts.append(t)
        labels.append(0)

    # Spam
    for _ in range(n_spam):
        t = random.choice(SPAM_SAMPLES)
        if augment:
            t = augment_text(t)
        texts.append(t)
        labels.append(1)

    # Toxic
    for _ in range(n_toxic):
        t = random.choice(TOXIC_SAMPLES)
        if augment and random.random() < 0.3:
            t = t + " " + random.choice([
                "I mean it.", "Don't ever contact me again.",
                "You'll regret this.", "I'm serious."
            ])
        texts.append(t)
        labels.append(2)

    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


if __name__ == "__main__":
    texts, labels = generate_dataset()
    counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
    print(f"Dataset: {len(texts)} samples | Ham:{counts[0]} Spam:{counts[1]} Toxic:{counts[2]}")
