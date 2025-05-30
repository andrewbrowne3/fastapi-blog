"""
Entity extraction utilities using spaCy for the customer service agent.
This module handles extraction of emails, phone numbers, and other personal details.
"""
import re
import spacy
from typing import Dict, List, Optional, Tuple, Any

# Load spaCy model - use the small model for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
    print("Loaded spaCy model successfully")
except OSError:
    print("SpaCy model not found. Installing en_core_web_sm...")
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    print("Installed and loaded spaCy model")

# Custom patterns for emails and phone numbers
EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
PHONE_PATTERN = r'(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}'

def name_matcher(text: str) -> Optional[str]:
    """
    Use spaCy's Matcher to identify name patterns in text.
    This can be more flexible than relying solely on NER.
    
    Args:
        text: Input text that might contain a name
        
    Returns:
        Extracted name or None
    """
    matcher = spacy.matcher.Matcher(nlp.vocab)
    
    # Pattern 1: First name + Last name
    pattern1 = [
        {"POS": "PROPN"},  # First name
        {"POS": "PROPN"}   # Last name
    ]
    
    # Pattern 2: Honorific + Name(s)
    pattern2 = [
        {"LOWER": {"IN": ["mr", "mrs", "ms", "miss", "dr", "prof"]}},
        {"TEXT": {"REGEX": "\.?"}, "OP": "?"},  # Optional period
        {"IS_SPACE": True, "OP": "?"},          # Optional space
        {"POS": "PROPN"}                        # Name
    ]
    
    # Pattern 3: My name is + Name(s)
    pattern3 = [
        {"LOWER": "my"},
        {"LOWER": {"IN": ["name", "names"]}},
        {"LOWER": {"IN": ["is", "are"]}},
        {"POS": "PROPN"},                      # First name
        {"POS": "PROPN", "OP": "?"}            # Optional last name
    ]
    
    # Pattern 4: I am + Name(s)
    pattern4 = [
        {"LOWER": {"IN": ["i", "this"]}},
        {"LOWER": {"IN": ["am", "is"]}},
        {"POS": "PROPN"},                      # First name
        {"POS": "PROPN", "OP": "?"}            # Optional last name
    ]
    
    # Pattern 5: This is + Name(s)
    pattern5 = [
        {"LOWER": "this"},
        {"LOWER": "is"},
        {"POS": "PROPN"},                      # First name
        {"POS": "PROPN", "OP": "?"}            # Optional last name
    ]
    
    # Add all patterns
    matcher.add("NAME_PATTERN", [pattern1], greedy="LONGEST")
    matcher.add("HONORIFIC_NAME", [pattern2], greedy="LONGEST")
    matcher.add("MY_NAME_IS", [pattern3], greedy="LONGEST")
    matcher.add("I_AM", [pattern4], greedy="LONGEST")
    matcher.add("THIS_IS", [pattern5], greedy="LONGEST")
    
    doc = nlp(text)
    matches = matcher(doc)
    
    # If we found matches
    if matches:
        # Sort by pattern (priority) and then by length (longer matches usually better)
        matches.sort(key=lambda x: (x[0], x[2]-x[1]), reverse=True)
        match_id, start, end = matches[0]
        
        # For "My name is" pattern, skip the intro words
        if nlp.vocab.strings[match_id] == "MY_NAME_IS":
            # Skip "My name is" part
            for token in doc[start:end]:
                if token.pos_ == "PROPN":
                    start = token.i
                    break
        
        # For "I am" or "This is" patterns, skip the intro words
        elif nlp.vocab.strings[match_id] in ["I_AM", "THIS_IS"]:
            # Skip "I am" or "This is" part
            for token in doc[start:end]:
                if token.pos_ == "PROPN":
                    start = token.i
                    break
        
        # Extract the name span
        name_span = doc[start:end]
        return name_span.text
    
    # If no matches, return None
    return None

def email_matcher(text: str) -> Optional[str]:
    """
    Use spaCy's Matcher to identify email addresses in text.
    More flexible than regex alone since it can consider context.
    
    Args:
        text: Input text that might contain an email address
        
    Returns:
        Extracted email or None
    """
    matcher = spacy.matcher.Matcher(nlp.vocab)
    
    # Pattern 1: Plain email detection
    # This is a simplified pattern that will match text that looks like an email
    # The actual validation will be done with regex after finding the potential match
    pattern1 = [
        {"TEXT": {"REGEX": "[a-zA-Z0-9._%+-]+"}},
        {"TEXT": {"REGEX": "@"}},
        {"TEXT": {"REGEX": "[a-zA-Z0-9.-]+"}}
    ]
    
    # Pattern 2: Email with introduction
    pattern2 = [
        {"LOWER": {"IN": ["email", "e-mail", "mail", "address"]}},
        {"LOWER": {"IN": ["is", "at", ":"]}},
        {"TEXT": {"REGEX": "[a-zA-Z0-9._%+-]+"}},
        {"TEXT": {"REGEX": "@"}},
        {"TEXT": {"REGEX": "[a-zA-Z0-9.-]+"}}
    ]
    
    # Pattern 3: "Contact me at" + email
    pattern3 = [
        {"LOWER": {"IN": ["contact", "reach", "email", "mail"]}},
        {"LOWER": {"IN": ["me", "us", "at", "via"]}},
        {"LOWER": {"IN": ["at", "via", "through", "using", ":"]}},
        {"TEXT": {"REGEX": "[a-zA-Z0-9._%+-]+"}},
        {"TEXT": {"REGEX": "@"}},
        {"TEXT": {"REGEX": "[a-zA-Z0-9.-]+"}}
    ]
    
    # Add all patterns
    matcher.add("EMAIL", [pattern1], greedy="LONGEST")
    matcher.add("EMAIL_INTRO", [pattern2], greedy="LONGEST")
    matcher.add("CONTACT_EMAIL", [pattern3], greedy="LONGEST")
    
    doc = nlp(text)
    matches = matcher(doc)
    
    # If we found matches
    if matches:
        # Sort by pattern (priority) and then by length (longer matches usually better)
        matches.sort(key=lambda x: (x[0], x[2]-x[1]), reverse=True)
        match_id, start, end = matches[0]
        
        # For patterns with introduction words, find the actual email part
        if nlp.vocab.strings[match_id] in ["EMAIL_INTRO", "CONTACT_EMAIL"]:
            email_text = ""
            for token in doc[start:end]:
                if "@" in token.text:
                    # Find the token with @ and its surrounding tokens
                    email_start = token.i
                    while email_start > start and re.match(r'[a-zA-Z0-9._%+-]', doc[email_start-1].text):
                        email_start -= 1
                    
                    email_end = token.i + 1
                    while email_end < end and re.match(r'[a-zA-Z0-9.-]', doc[email_end].text):
                        email_end += 1
                    
                    email_text = doc[email_start:email_end].text
                    break
            
            # If we found an email, validate with regex
            if email_text and "@" in email_text:
                email_matches = re.findall(EMAIL_PATTERN, email_text)
                if email_matches:
                    return email_matches[0]
        else:
            # For direct email pattern, extract and validate
            email_text = doc[start:end].text
            email_matches = re.findall(EMAIL_PATTERN, email_text)
            if email_matches:
                return email_matches[0]
    
    # If no matches or validation failed, use regex on the full text as fallback
    email_matches = re.findall(EMAIL_PATTERN, text)
    if email_matches:
        return email_matches[0]
    
    return None

def phone_matcher(text: str) -> Optional[str]:
    """
    Use spaCy's Matcher to identify phone numbers in text.
    More flexible than regex alone since it can consider context.
    
    Args:
        text: Input text that might contain a phone number
        
    Returns:
        Extracted phone number or None
    """
    matcher = spacy.matcher.Matcher(nlp.vocab)
    
    # Pattern 1: Sequence of numbers with possible formatting
    pattern1 = [
        {"IS_DIGIT": True, "LENGTH": {"IN": [3]}},  # Area code
        {"TEXT": {"REGEX": "[- .()]?"}, "OP": "?"},  # Optional separator
        {"IS_DIGIT": True, "LENGTH": {"IN": [3]}},  # Exchange code
        {"TEXT": {"REGEX": "[- .()]?"}, "OP": "?"},  # Optional separator
        {"IS_DIGIT": True, "LENGTH": {"IN": [4]}}    # Subscriber number
    ]
    
    # Pattern 2: Phone number with country code
    pattern2 = [
        {"TEXT": {"REGEX": "\\+?\\d{1,3}"}},        # Country code with optional +
        {"TEXT": {"REGEX": "[- .()]?"}, "OP": "?"},  # Optional separator
        {"IS_DIGIT": True, "LENGTH": {"IN": [3]}},  # Area code
        {"TEXT": {"REGEX": "[- .()]?"}, "OP": "?"},  # Optional separator
        {"IS_DIGIT": True, "LENGTH": {"IN": [3]}},  # Exchange code
        {"TEXT": {"REGEX": "[- .()]?"}, "OP": "?"},  # Optional separator
        {"IS_DIGIT": True, "LENGTH": {"IN": [4]}}    # Subscriber number
    ]
    
    # Pattern 3: Phone with introduction
    pattern3 = [
        {"LOWER": {"IN": ["phone", "number", "cell", "mobile", "telephone", "call"]}},
        {"LOWER": {"IN": ["is", "at", "number", ":"]}},
        {"TEXT": {"REGEX": "[0-9()+\\-. ]+"}}  # Any sequence with digits and formatting
    ]
    
    # Add all patterns
    matcher.add("PHONE", [pattern1], greedy="LONGEST")
    matcher.add("PHONE_COUNTRY", [pattern2], greedy="LONGEST")
    matcher.add("PHONE_INTRO", [pattern3], greedy="LONGEST")
    
    doc = nlp(text)
    matches = matcher(doc)
    
    # If we found matches
    if matches:
        # Sort by pattern (priority) and then by length (longer matches usually better)
        matches.sort(key=lambda x: (x[0], x[2]-x[1]), reverse=True)
        match_id, start, end = matches[0]
        
        # For patterns with introduction words, find the actual phone part
        if nlp.vocab.strings[match_id] == "PHONE_INTRO":
            # Look for numeric sequences in the match
            phone_text = ""
            for i in range(start, end):
                token = doc[i]
                if any(c.isdigit() for c in token.text):
                    # Find a sequence with digits
                    digits_start = i
                    while digits_start < end:
                        if any(c.isdigit() for c in doc[digits_start].text):
                            break
                        digits_start += 1
                    
                    digits_end = digits_start
                    while digits_end < end:
                        digits_end += 1
                    
                    phone_text = doc[digits_start:digits_end].text
                    break
            
            # Extract and validate the phone number if found
            if phone_text:
                phone_matches = re.findall(PHONE_PATTERN, phone_text)
                if phone_matches:
                    return phone_matches[0]
        else:
            # For direct phone patterns, extract the full match
            phone_text = doc[start:end].text
            phone_matches = re.findall(PHONE_PATTERN, phone_text)
            if phone_matches:
                return phone_matches[0]
            else:
                # If regex doesn't match, attempt to clean and format the text
                digits_only = ''.join(c for c in phone_text if c.isdigit())
                if len(digits_only) >= 10:  # Minimum 10 digits for a valid phone
                    # Format as (123) 456-7890
                    if len(digits_only) == 10:
                        return f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
                    elif len(digits_only) > 10:  # Has country code
                        country = digits_only[:-10]
                        return f"+{country} ({digits_only[-10:-7]}) {digits_only[-7:-4]}-{digits_only[-4:]}"
    
    # If no matches or validation failed, use regex on the full text as fallback
    phone_matches = re.findall(PHONE_PATTERN, text)
    if phone_matches:
        return phone_matches[0]
    
    return None

def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extract relevant entities from user input using spaCy and regex.
    
    Args:
        text: The user's input text
        
    Returns:
        dict: Dictionary with extracted entities
    """
    # Process text with spaCy
    doc = nlp(text)
    
    # Initialize results
    result = {
        "email": None,
        "phone": None,
        "person": None,
        "org": None,
        "date": None,
        "has_contact_info": False,
        "extracted_entities": []
    }
    
    # Extract standard entities
    for ent in doc.ents:
        result["extracted_entities"].append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
        
        # Store specific entity types
        if ent.label_ == "PERSON":
            result["person"] = ent.text
        elif ent.label_ == "ORG":
            result["org"] = ent.text
        elif ent.label_ == "DATE":
            result["date"] = ent.text
    
    # If no person found using NER, try the matcher approach
    if not result["person"]:
        name = name_matcher(text)
        if name:
            result["person"] = name
    
    # Use matchers for emails and phone numbers
    # Try pattern-based matcher first, then fall back to regex if needed
    email = email_matcher(text)
    if email:
        result["email"] = email
        result["has_contact_info"] = True
    
    phone = phone_matcher(text)
    if phone:
        result["phone"] = phone
        result["has_contact_info"] = True
    
    return result

def analyze_response_for_contact_info(response: str) -> Dict[str, Any]:
    """
    Analyze user response to determine if it contains contact information
    and what type of follow-up might be needed.
    
    Args:
        response: User's response to a question about contact information
        
    Returns:
        dict: Analysis results with flags for contact info and follow-up needs
    """
    entities = extract_entities(response)
    
    # Determine response type based on content
    is_affirmative = any(word in response.lower() for word in ["yes", "yeah", "sure", "okay", "fine", "yep"])
    is_negative = any(word in response.lower() for word in ["no", "nope", "don't", "not", "won't", "can't"])
    
    # Analyze what we found and what's missing
    analysis = {
        "has_contact_info": entities["has_contact_info"],
        "is_affirmative": is_affirmative,
        "is_negative": is_negative,
        "needs_email_followup": is_affirmative and not entities["email"],
        "needs_phone_followup": is_affirmative and not entities["phone"],
        "email": entities["email"],
        "phone": entities["phone"]
    }
    
    return analysis

def determine_contact_followup(analysis: Dict[str, Any]) -> str:
    """
    Determine what type of follow-up is needed based on the analysis.
    
    Args:
        analysis: Result from analyze_response_for_contact_info
        
    Returns:
        str: The type of follow-up needed ("email", "phone", "none", "both")
    """
    if analysis["is_negative"]:
        return "none"
    
    if analysis["has_contact_info"]:
        return "none"
    
    if analysis["needs_email_followup"] and analysis["needs_phone_followup"]:
        return "both"
    elif analysis["needs_email_followup"]:
        return "email"
    elif analysis["needs_phone_followup"]:
        return "phone"
    
    return "none"

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment and intent from user feedback using spaCy.
    
    Args:
        text: User's feedback text
        
    Returns:
        dict: Analysis results with sentiment and detected intents
    """
    doc = nlp(text.lower())
    
    # Initialize sentiment analysis
    result = {
        "sentiment": "neutral",  # Default sentiment
        "confidence": 0.5,  # Default confidence
        "is_negative": False,
        "is_positive": False,
        "contains_question": False,
        "wants_followup": False,
        "wants_to_end": False,
        "detected_topics": [],
    }
    
    # Negation words that might indicate dissatisfaction
    negation_words = ["not", "no", "never", "neither", "nor", "none", "nothing", "nowhere"]
    
    # Positive sentiment words
    positive_words = ["good", "great", "excellent", "helpful", "thanks", "thank", "appreciate", 
                     "useful", "clear", "perfect", "wonderful", "awesome", "satisfied", "happy"]
    
    # Negative sentiment words
    negative_words = ["bad", "poor", "terrible", "unhelpful", "useless", "wrong", "incorrect", 
                     "confused", "disappointed", "frustrating", "waste", "difficult", "unclear"]
    
    # Follow-up indicators
    followup_words = ["more", "another", "also", "additional", "explain", "clarify", 
                     "elaborate", "detail", "question", "yes", "yeah"]
    
    # Ending conversation indicators
    ending_words = ["bye", "goodbye", "end", "stop", "quit", "finished", "done", "complete", "that's all"]
    
    # Get tokens without stopwords (except negations which are important)
    tokens = [token.text for token in doc if not token.is_stop or token.text in negation_words]
    
    # Check for question marks or question words
    question_words = ["what", "how", "why", "when", "where", "who", "which"]
    contains_question = "?" in text or any(word in text.lower().split() for word in question_words)
    result["contains_question"] = contains_question
    
    # Count sentiment indicators
    positive_count = sum(1 for word in positive_words if word in tokens)
    negative_count = sum(1 for word in negative_words if word in tokens)
    
    # Check for negated positive words (e.g., "not helpful")
    for i, token in enumerate(tokens[:-1]):
        if token in negation_words and i+1 < len(tokens) and tokens[i+1] in positive_words:
            positive_count -= 1
            negative_count += 1
    
    # Determine overall sentiment
    if negative_count > positive_count:
        result["sentiment"] = "negative"
        result["is_negative"] = True
        result["confidence"] = 0.5 + (negative_count - positive_count) * 0.1
    elif positive_count > negative_count:
        result["sentiment"] = "positive"
        result["is_positive"] = True
        result["confidence"] = 0.5 + (positive_count - negative_count) * 0.1
    
    # Cap confidence at 0.9
    result["confidence"] = min(result["confidence"], 0.9)
    
    # Determine if they want follow-up
    if any(word in tokens for word in followup_words) or contains_question:
        result["wants_followup"] = True
    
    # Determine if they want to end the conversation
    if any(word in tokens for word in ending_words) or (len(tokens) <= 3 and "no" in tokens):
        result["wants_to_end"] = True
    
    # Check for domain-specific topics
    domain_topics = {
        "admissions": ["admission", "apply", "application", "enroll", "enrollment"],
        "financial": ["tuition", "cost", "scholarship", "financial", "aid", "loan", "price", "afford"],
        "academic": ["course", "program", "curriculum", "class", "study", "academic", "faculty"],
        "technical": ["website", "portal", "login", "password", "account", "access", "online"],
    }
    
    for topic, keywords in domain_topics.items():
        if any(keyword in tokens for keyword in keywords):
            result["detected_topics"].append(topic)
    
    return result

# For testing
if __name__ == "__main__":
    test_inputs = [
        "My email is john.doe@example.com",
        "You can reach me at 555-123-4567",
        "Yes, I'm happy to provide my contact info",
        "No, I prefer not to share my details",
        "My name is Jane Smith and my phone number is (123) 456-7890",
        "Contact me at jane@example.com or 987-654-3210"
    ]
    
    # Test sentiment analysis
    sentiment_tests = [
        "This was really helpful, thank you.",
        "I'm not satisfied with this answer.",
        "Can you tell me more about the nursing program?",
        "No, that's all I needed.",
        "The information wasn't very clear.",
        "Yes, I'd like to know about tuition costs."
    ]
    
    print("\nTesting sentiment analysis:")
    for test in sentiment_tests:
        sentiment = analyze_sentiment(test)
        print(f"\nInput: {test}")
        print(f"Sentiment: {sentiment['sentiment']} (confidence: {sentiment['confidence']:.2f})")
        print(f"Wants follow-up: {sentiment['wants_followup']}")
        print(f"Wants to end: {sentiment['wants_to_end']}")
        if sentiment['detected_topics']:
            print(f"Topics: {', '.join(sentiment['detected_topics'])}")
    
    for test in test_inputs:
        print(f"\nInput: {test}")
        entities = extract_entities(test)
        analysis = analyze_response_for_contact_info(test)
        followup = determine_contact_followup(analysis)
        
        print(f"Email: {entities['email']}")
        print(f"Phone: {entities['phone']}")
        print(f"Person: {entities['person']}")
        print(f"Has contact info: {entities['has_contact_info']}")
        print(f"Needs followup: {followup}") 