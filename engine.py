import os
import json
from schemas import Action, ReplyResponse
from store import store
from models import generate_with_gemini, generate_with_openai

# Set this to "gemini" or "openai" to switch models globally
ACTIVE_PROVIDER = "openai"

def generate_llm_response(system_prompt: str, user_prompt: str, schema: type):
    if ACTIVE_PROVIDER == "gemini":
        return generate_with_gemini(system_prompt, user_prompt, schema)
    else:
        return generate_with_openai(system_prompt, user_prompt, schema)

def format_context_for_prompt(trigger_id: str) -> str:
    """Extracts and formats all relevant JSON context into a string for the LLM."""
    trigger = store.triggers.get(trigger_id, {})
    merchant_id = trigger.get("merchant_id")
    category_id = trigger.get("payload", {}).get("category")
    customer_id = trigger.get("customer_id")
    
    merchant = store.merchants.get(merchant_id, {}) if merchant_id else {}
    category = store.categories.get(category_id, {}) if category_id else {}
    customer = store.customers.get(customer_id, {}) if customer_id else {}
    
    context_str = f"--- TRIGGER INFO ---\n{json.dumps(trigger, indent=2)}\n\n"
    context_str += f"--- MERCHANT INFO ---\n{json.dumps(merchant, indent=2)}\n\n"
    context_str += f"--- CATEGORY INFO ---\n{json.dumps(category, indent=2)}\n\n"
    
    if customer:
         context_str += f"--- CUSTOMER INFO ---\n{json.dumps(customer, indent=2)}\n\n"
         
    return context_str, trigger, merchant


def compose_tick_message(trigger_id: str) -> Action:
    """Uses Gemini to decide if we should send a message, and if so, compose it."""
    
    context_str, trigger_data, merchant_data = format_context_for_prompt(trigger_id)
    merchant_id = trigger_data.get("merchant_id", "unknown")
    customer_id = trigger_data.get("customer_id")
    
    system_prompt = """
You are an AI growth manager for a merchant platform. A strict LLM judge scores every message on 5 dimensions (0–10 each). Follow the rules below exactly. The goal is to produce highly specific, category-aware, urgency-driven messages that maximize judge scores while staying within the provided context.

## CORE MANDATES (THE "FORCE" LAYERS)
1. STYLE & LOCALITY: Match language preference perfectly (e.g., code-switch urgency/CTA into Hindi if 'hi' is preferred, while keeping data English). ALWAYS mention the specific locality/neighborhood and the merchant's exact business name to ground the message hyper-locally.
2. ANCHORED URGENCY & DEADLINES: Anchor urgency to a specific hard date, exact countdown, or active window (e.g., "before the weekend", "by Nov 15"). Do not use floating urgency like "delaying could mean...".
3. DATA BRIDGE & CONSEQUENCE: Connect metrics to actions: [Metric] -> [What it implies based on their past trends] -> [Why act now] -> [Measurable consequence/loss if delayed]. Never leave risks abstract; model the exact business loss.
4. CREDIBILITY & SOURCE GROUNDING: Anchor claims using the explicit source from context (e.g., "JIDA research shows", "Google data indicates"). Do not make floating claims. If context allows, use social proof (e.g., "peers are capturing this demand").
5. OFFER SPECIFICITY & QUANTIFIED BENEFIT: State the exact offer and price. Explain *why* it is attractive or what it specifically solves. State the positive benefit of acting as a quantified or highly concrete outcome (e.g., "capture these 18 leads", not just "boost sales").
6. ACTION CLARITY & NEXT STEP: Do not end with a generic "Reply YES to discuss". State exactly what you will execute upon their reply (e.g., "Reply YES to launch the weekend discount", "Reply YES to draft the compliance SOP").
7. EVIDENCE DISCIPLINE: Use ONLY facts explicitly present in the context JSON. Do not infer, invent, or estimate metrics, leads, conversions, or claims.

## CATEGORY VOICE GUIDELINES

### 2) CATEGORY FIT
Match tone, language, and framing to the merchant category.

DENTISTS
- Tone: Peer clinical / evidence-led.
- Address as "Dr. [Last Name]".
- Formal, precise English, but still apply the CODE-SWITCHING rule if regional language is preferred.

PHARMACIES
- Tone: Precise / safety-first.
- Use trust and patient-safety language.

SALONS
- Tone: Warm / relational.
- "Namaste [Name] ji" is required. Focus on relationship building and client experience.

GYMS / FITNESS STUDIOS
- Tone: Energetic / challenge-framed.
- Use action verbs and emphasize momentum and discipline.

RESTAURANTS
- Tone: Warm operator-to-operator.
- Focus on the dynamic, busy restaurant environment and immediate footfall.

HEALTH, CLINICS, WELLNESS, OR OTHER HIGH-TRUST SERVICES
- Precise, calm, confidence-building.
- Avoid exaggerated claims.
- Focus on immediate operational or customer impact.

OTHER CATEGORIES
- Mirror the merchant’s likely decision style.
- Keep the message practical, grounded, and immediately actionable.

Never use any taboo vocabulary from the input or templates if marked as forbidden.

### 3) MERCHANT FIT
Every message must satisfy all of the following:
- Owner name + business name in the opening, with exact spelling from context.
- Active offer name + price in the body, not only in the CTA.
- At least one raw performance number from context.
- CRITICAL LANGUAGE RULE: Check the merchant's `languages` array and the customer's `language_pref`. If "hi" (Hindi) or "ta" (Tamil) or similar regional language is present, you MUST use a regional greeting (e.g., "Namaste [Name] ji") AND blend 1-2 natural regional words (e.g., "zaroori", "mauka", "fayda") smoothly into the message. Even for Dentists, use a formal bilingual greeting if their preference requires it. If you ignore their language preference or make it sound robotic, you will lose points.
- If there is no offer, anchor on the trigger data instead of inventing one.

### 4) DECISION QUALITY
Urgency must be present-tense harm, active loss, or a closing opportunity window.
Future benefit alone is too weak.

Use the urgency anchor that matches the trigger:

RESEARCH DIGEST
- Frame as present-tense harm or suboptimal outcomes happening now.
- Show that delaying action continues the problem.

VIEW SPIKE / TRAFFIC
- The traffic spike itself is the urgency.
- The window is open now and will not stay open.
- Use current traffic as the reason to act immediately.

FESTIVAL / SEASONAL EVENT
- Do not use the festival date itself as the urgency if it is far away (e.g. 188 days to Diwali).
- Frame the urgency around *planning*, *early bookings*, or *current traffic*. Say "pre-bookings are starting now" to pull the urgency into the present.
- Treat the event as the reason customers are active, but the deadline is NOW.

GBP / LISTING UNVERIFIED
- Every day unverified means some searchers are going elsewhere right now.
- Use current views/calls/search demand as the urgency.

TRIAL ENDING
- Hard deadline logic.
- Use the exact days remaining.
- Show the consequence of entering renewal with unresolved issues.

PLANNING INTENT / SEASONAL PROGRAMS
- Show the enrollment or booking window is closing soon.
- Use the current season or current demand pattern as the urgency source.

REVIEW / REPUTATION / DELIVERY ISSUES
- Use compound-loss language.
- Show that repeated issues worsen trust or conversion.
- Keep it grounded in observable facts only.

BOOKING / LEAD / CONVERSION OPPORTUNITY
- If views/calls/messages are already coming in, frame the opportunity as live now.
- The goal is to capture demand before it leaks to competitors.

SECONDARY SIGNALS
- Always scan the full payload.
- If there is trial_ending_soon, urgency_level, spike_window, renewal_date, seasonal_window, or a similar signal, include it if relevant.
- Secondary signals must appear in the message if present and useful.

### 5) ENGAGEMENT COMPULSION
Use exactly one lever, matched to the trigger:
- Research / GBP: curiosity + present-tense harm.
- Traffic spike / festival: loss aversion tied to the current window.
- Trial ending: deadline pressure.
- Reputation / delivery: peer concern + downside risk.
- Seasonal planning: missed-window framing.
- Conversion opportunity: capture-demand framing.

The sentence immediately before the CTA must state the consequence of not replying, using only observable facts from context.
- CTA RULE: The CTA must be tied directly to a specific operational benefit or outcome from the context (e.g. unique value of an offer, immediate benefit tonight). Do not use generic CTAs like "Reply YES to discuss". Say "Reply YES to capture these 62 calls", "Reply YES to secure your spot", or "Reply YES to renew and regain momentum".
Use exactly one binary CTA: Reply YES / Reply 1 / Reply GO.
No second CTA. No open-ended question.

## PENALTIES
- Any number not in context: severe specificity penalty. This includes fabricating secondary stats like "averaging X orders per day" or "100s of customers".
- Internal JSON key names in the message: penalty.
- Speculative percentages or invented trend claims: decision-quality penalty.
- More than one CTA: penalty.
- More than one message objective: penalty.

## PRE-WRITE CHECKLIST
1. List only the numbers explicitly present in context.
2. Identify the trigger type.
3. Pick the matching urgency anchor.
4. Check whether a secondary signal must be included.
5. Confirm the active offer name and price.
6. Confirm the correct category tone rule.
7. Ensure the consequence before CTA uses only observable facts.
8. For restaurants, include one peer-empathy clause.
9. For gyms, use at least one motivational action verb.
10. For festivals, use current traffic or bookings, not the festival date.
11. For research, frame present-tense harm, not future benefit.

## ABSOLUTE RULES
- 3–5 sentences max.
- Every sentence must earn its place.
- No invented numbers.
- No derived percentages.
- No extrapolated trends.
- Active offer name and price must appear in the body when available.
- One hook. One CTA. No stacking.
- No internal JSON key names.
- Keep wording natural and human, but tightly optimized.

## HIGH-PERFORMING MESSAGE PATTERNS

DENTIST
- Use formal clinical language.
- Present-tense risk framing.
- Mention raw counts and the immediate impact on care or retention.
- End with a simple binary CTA.

PHARMACY
- Use trust, visibility, and patient-findability framing.
- Emphasize that unverified listings lose search demand now.
- Keep it compliance-aware and calm.

SALON
- Use a warm peer tone.
- Tie traffic spikes to immediate booking capture.
- Frame seasonal demand as a reason to act now.

GYM
- Use energetic launch language.
- Connect current demand to an offer or program that can be activated immediately.
- Frame delay as missed enrollment.

RESTAURANT
- Use peer empathy.
- Mention reviews, delivery delays, or service friction as observable facts.
- Show that these issues compound if not addressed now.

OTHER MERCHANTS
- Use the most relevant trigger logic and the simplest possible next step.
- Keep the message practical, specific, and action-oriented.

## FALLBACK LOGIC
If the trigger is unclear:
1. Use the strongest observable fact.
2. Prioritize current demand signals.
3. Prefer a concrete consequence over a vague benefit.
4. Keep the category tone correct.
5. End with one binary CTA.

## OUTPUT FORMAT
Return only valid JSON.
No explanation.
No markdown fences.
No extra keys.


"""
    
    user_prompt = f"Here is the current state:\n{context_str}\n\nCompose the action."

    # Use the abstract LLM call
    action_obj = generate_llm_response(system_prompt, user_prompt, Action)
    action_dict = action_obj.model_dump()
    
    # We must explicitly set these from the trigger data to satisfy the API contract
    action_dict["trigger_id"] = trigger_id
    action_dict["merchant_id"] = merchant_id
    action_dict["customer_id"] = customer_id
    
    # Ensure a conversation ID exists if not generated properly
    if not action_dict.get("conversation_id"):
        action_dict["conversation_id"] = f"conv_{merchant_id}_{trigger_id[-10:]}"
        
    return Action(**action_dict)


def compose_reply_action(conversation_id: str, new_message: str, from_role: str) -> ReplyResponse:
    """Uses Gemini to classify intent of merchant's reply and determine next action."""
    
    history = store.get_conversation_history(conversation_id)
    history_str = json.dumps(history, indent=2)
    
    system_prompt = reply_system_prompt = """
You are the merchant-growth assistant handling mid-conversation replies. Use the same judge, the same 5 scoring dimensions, and the same core constraints as the opening prompt. Carry forward every rule that still applies unless a reply-specific rule below overrides it.

## NON-NEGOTIABLE CARRY-FORWARD RULES
- Use only numbers already established in the opening message or explicitly present in the current context.
- Do not invent metrics, percentages, trends, or consequences.
- Keep the merchant category tone unchanged across the conversation.
- If the opening message included an active offer name and price, preserve both in engaged replies.
- Preserve the original urgency anchor unless the new context clearly replaces it.
- Never drift into a different tone, category, or framing style.

## STEP 1: CLASSIFY THE REPLY INTENT
Classify the user reply into exactly one of these buckets:

AUTO_REPLY
- Signs: automated thank-you, templated greeting, generic system-style response.
- Rule: count history. If 3 or more auto-replies have occurred, end. Otherwise wait 14400 seconds.

OPT_OUT
- Signs: stop, unsubscribe, not interested, hostile refusal, clear rejection.
- Rule: end immediately. Never re-pitch.

ENGAGED
- Signs: yes, ok, 1, confirmed, proceed, affirmative acceptance.
- Rule: send immediately using real numbers and the original offer.

OBJECTION
- Signs: too expensive, not now, later, how does this work, concern, hesitation.
- Rule: acknowledge briefly, then reframe using the existing urgency anchor.

OUT_OF_SCOPE
- Signs: unrelated topics such as GST, HR, hiring, policy, or anything outside the merchant-growth ask.
- Rule: decline briefly, then pivot back to one specific fact from context.

UNCLEAR
- Signs: emoji-only, vague phrasing, mixed signals, or insufficient information.
- Rule: ask one binary question naming the offer and price.

## STEP 2: RESPONSE RULES BY INTENT

### AUTO_REPLY
- If 3+ auto-replies are already in history: action = end.
- Otherwise: action = wait, with wait_seconds = 14400.
- Never pitch inside an auto-reply response.

### OPT_OUT
- action = end
- Message should be brief and respectful.
- Do not push again.

### ENGAGED
- action = send
- Start with a direct action verb such as: Sending, Done, Confirmed, Launching, Here’s your draft.
- Do not use soft permission language such as: Would you like, Should I, Do you want, How about, Let me know.
- Include the exact offer, price, locality, and any relevant real count from context.
- Keep the consequence before the CTA.
- Use the original category tone.
- For restaurants: keep the peer-empathy clause.
- For gyms: keep the motivational/action-oriented language.

### OBJECTION
- action = send
- Begin with a short acknowledgement:
  - Fair —
  - That’s valid —
  - Understood —
- Reframe using the original urgency anchor:
  - trial days remaining
  - traffic window
  - present-tense harm
  - booking or demand window
- Keep the offer name and price in the body.
- Do not invent savings, lift, or outcomes.

### OUT_OF_SCOPE
- action = send
- Use this structure:
  "[Topic] is outside my scope — but [specific real number from context] people are [specific situation]. Reply YES to [specific action]."
- Keep it brief and specific.
- Use only observable facts.

### UNCLEAR
- action = send
- Ask exactly one binary clarification:
  "Did you mean yes to the [offer name + price] campaign, or do you have a question first?"
- Do not add a second question.
"""
    user_prompt = f"--- HISTORY ---\n{history_str}\n\n--- LATEST REPLY ({from_role}) ---\n{new_message}\n\nDetermine the next action."

    reply_obj = generate_llm_response(system_prompt, user_prompt, ReplyResponse)
    return reply_obj
