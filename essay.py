import streamlit as st
import random
import re

st.set_page_config(page_title="PTE Academic Write Essay", page_icon="📝", layout="centered")

# --- PTE Write Essay Prompt Bank ---
ESSAY_PROMPT_BANK = [
    {
        "prompt": "In the modern world, the use of mobile phones and computers has changed the way people communicate. Do the advantages of this development outweigh the disadvantages?",
        "keywords": ["communication", "mobile", "computer", "advantage", "disadvantages", "advantages", "disadvantage", "technology", "phone", "social"]
    },
    {
        "prompt": "Some people believe that university education should be free for everyone, regardless of their background. To what extent do you agree or disagree with this opinion?",
        "keywords": ["university", "education", "free", "tuition", "students", "agree", "disagree", "background", "funding"]
    },
    {
        "prompt": "Climate change is one of the most pressing issues facing the world today. Who should be responsible for mitigating it: governments, corporations, or individuals?",
        "keywords": ["climate", "change", "environment", "government", "corporation", "individual", "responsibility", "pollution", "corporations", "governments", "individuals"]
    },
    {
        "prompt": "As cities continue to expand, green spaces are often replaced by commercial buildings. Discuss the problems associated with this trend and suggest some potential solutions.",
        "keywords": ["city", "urban", "green", "space", "park", "building", "problem", "solution", "environment", "buildings", "cities", "problems", "solutions"]
    },
    {
        "prompt": "Mass media, including television, radio, and newspapers, has a powerful influence on shaping public opinion. Is this influence more positive or negative?",
        "keywords": ["media", "television", "influence", "public", "opinion", "positive", "negative", "news", "information"]
    }
]

# Preserve target essay prompt across page re-renders
if "current_essay" not in st.session_state:
    st.session_state.current_essay = random.choice(ESSAY_PROMPT_BANK)

st.title("📝 PTE Academic Write Essay Lab")
st.caption("Writing Module • Official Pearson 15-Point Matrix & GSE Scoring")

# --- Info & Guidelines Box ---
with st.expander("ℹ️ Official PTE Essay Rules & Scoring Criteria", expanded=False):
    st.markdown("""
    Your response is judged on an automated **15-point enabling scale**, then mapped to the global **10-90 score**:
    * **Form (0-2 marks):** Length must be strictly between **200–300 words** for full credit. *Crucial rule: If your essay is less than 120 words or more than 380 words, you score 0 overall.*
    * **Content (0-3 marks):** Relevancy to the topic. If content scores 0, the engine stops grading completely.
    * **Development, Structure & Coherence (0-2 marks):** Clear paragraph structures and logical transitions.
    * **Grammar & Linguistic Range (0-4 marks):** Structural variety, complex sentences, and accuracy.
    * **Vocabulary Range (0-2 marks):** Use of academic words and avoiding over-repetition.
    * **Spelling (0-2 marks):** Level of typos and consistency.
    """)

# --- Prompt Display ---
st.info(f"**Exam Prompt:**\n\n\"{st.session_state.current_essay['prompt']}\"")

# --- Control Panel Bar ---
if st.button("🔄 Next Essay Prompt"):
    st.session_state.current_essay = random.choice(ESSAY_PROMPT_BANK)
    if "essay_input" in st.session_state:
        st.session_state.essay_input = ""
    st.rerun()

st.markdown("---")

# --- Essay Text Input Window ---
if "essay_input" not in st.session_state:
    st.session_state.essay_input = ""

def update_word_count():
    st.session_state.essay_input = st.session_state.new_essay_text

essay_text = st.text_area(
    "Type your essay response below (Recommended: 20-minute limit):", 
    height=350, 
    placeholder="Write your response here... (Press Ctrl+Enter or click outside to update your live metrics)",
    key="new_essay_text",
    on_change=update_word_count,
    value=st.session_state.essay_input
)

# Live Word Counter Layout
words = essay_text.split()
word_count = len(words)

if word_count == 0:
    st.caption("Word Count: 0 words")
elif 200 <= word_count <= 300:
    st.success(f"🟩 Word Count: {word_count} words (Optimal Range)")
elif (120 <= word_count < 200) or (301 <= word_count <= 380):
    st.warning(f"🟨 Word Count: {word_count} words (Acceptable with form penalty)")
else:
    st.error(f"🟥 Word Count: {word_count} words (Critical Penalty: Will score 0 in Pearson Exam)")

# --- Evaluation Engine ---
if st.button("📊 Grade and Analyze Essay", type="primary", use_container_width=True):
    if word_count < 5:
        st.error("Please enter a valid essay response before submitting for analytical evaluation.")
    else:
        with st.spinner("PTE AI Grading Engine analyzing text structures..."):
            
            grammar_errors = []
            spelling_errors = []
            
            # 1. FORM SCORING (Max 2)
            if 200 <= word_count <= 300:
                form_score = 2
            elif (120 <= word_count < 200) or (301 <= word_count <= 380):
                form_score = 1
            else:
                form_score = 0
                
            # 2. CONTENT SCORING (Max 3)
            matched_keywords = [w for w in st.session_state.current_essay['keywords'] if w in essay_text.lower()]
            match_ratio = len(matched_keywords) / len(st.session_state.current_essay['keywords'])
            
            if match_ratio >= 0.4:
                content_score = 3
            elif match_ratio >= 0.2:
                content_score = 2
            elif match_ratio > 0:
                content_score = 1
            else:
                content_score = 0

            # Hard Rule Override
            if form_score == 0 or content_score == 0:
                st.error("⚠️ Automated Test Failure Event: The essay failed either basic Word Count constraints or missed the prompt topic. Under PTE guidelines, the engine stops further analysis.")
                st.metric("OVERALL SCORE", "10 / 90")
                
                st.markdown("### Detailed Structural Diagnostic Breakdown")
                st.write(f"**Form Requirement score:** {form_score} / 2")
                st.write(f"**Content Relevance score:** {content_score} / 3")
            else:
                # 3. DEVELOPMENT, STRUCTURE & COHERENCE (Max 2)
                paragraphs = [p.strip() for p in essay_text.split('\n') if p.strip()]
                num_paragraphs = len(paragraphs)
                
                cohesive_devices = ["however", "furthermore", "in addition", "consequently", "therefore", "on the other hand", "in conclusion", "to sum up", "firstly", "secondly"]
                found_cohesives = [w for w in cohesive_devices if w in essay_text.lower()]
                
                if num_paragraphs >= 4 and len(found_cohesives) >= 3:
                    dev_score = 2
                elif num_paragraphs >= 3 and len(found_cohesives) >= 1:
                    dev_score = 1
                else:
                    dev_score = 0

                # 4. GRAMMAR (Max 2)
                sentences = re.split(r'[.!?]+', essay_text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                for idx, s in enumerate(sentences):
                    if s and not s[0].isupper():
                        snippet = s[:40] + "..." if len(s) > 40 else s
                        grammar_errors.append(f"Sentence {idx+1} does not start with a capital letter: \"{snippet}\"")
                
                if len(grammar_errors) == 0 and len(sentences) >= 6:
                    grammar_score = 2
                elif len(grammar_errors) <= 2 and len(sentences) >= 4:
                    grammar_score = 1
                else:
                    grammar_score = 0

                # 5. GENERAL LINGUISTIC RANGE (Max 2)
                sentence_lengths = [len(s.split()) for s in sentences]
                has_variety = max(sentence_lengths) - min(sentence_lengths) > 8 if sentence_lengths else False
                
                # Check for complex clause markers to evaluate sentence diversity
                complex_markers = ["although", "because", "while", "whereas", "since", "provided that", "even though", "which", "who"]
                found_complex = [m for m in complex_markers if m in essay_text.lower()]
                
                if has_variety and len(sentences) >= 6 and len(found_complex) >= 2:
                    linguistic_score = 2
                elif len(sentences) >= 4:
                    linguistic_score = 1
                else:
                    linguistic_score = 0

                # 6. VOCABULARY RANGE (Max 2)
                unique_words = set([w.lower().strip(".,!?\"()") for w in words])
                lexical_diversity = len(unique_words) / word_count if word_count > 0 else 0
                
                if lexical_diversity > 0.4:
                    vocab_score = 2
                elif lexical_diversity > 0.25:
                    vocab_score = 1
                else:
                    vocab_score = 0

                # 7. SPELLING (Max 2) — High-Stability Structural Typo Scanner
                for original_w in words:
                    w = original_w.lower().strip(".,!?\"();:[]{}*")
                    if len(w) <= 3 or not w.isalpha():
                        continue
                    if re.search(r'(.)\1\1', w):
                        spelling_errors.append(original_w.strip(".,!?\"();:"))
                        continue
                    vowels = set("aeiouy")
                    vowel_count = sum(1 for char in w if char in vowels)
                    if vowel_count == 0:
                        spelling_errors.append(original_w.strip(".,!?\"();:"))
                
                spelling_errors = list(set(spelling_errors))
                
                if len(spelling_errors) == 0:
                    spelling_score = 2
                elif len(spelling_errors) <= 2:
                    spelling_score = 1
                else:
                    spelling_score = 0

                # --- SCORE CALCULATIONS & MAPPING TO GSE (10 - 90) ---
                total_trait_points = (
                    content_score + form_score + dev_score + 
                    grammar_score + linguistic_score + vocab_score + spelling_score
                )
                
                percentage_earned = total_trait_points / 15
                overall_score = round(10 + (80 * percentage_earned))
                overall_score = max(10, min(90, overall_score))
                
                # --- UI Scorecard Display ---
                st.markdown("### 📊 Your Performance Evaluation Profile")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("OVERALL SCORE", f"{overall_score} / 90")
                with col2:
                    if overall_score >= 79:
                        st.success("Target Level: C1/C2 Professional Mastery (Superior Performance)")
                    elif overall_score >= 65:
                        st.info("Target Level: B2 Advanced Fluency (Proficient Performance)")
                    elif overall_score >= 50:
                        st.warning("Target Level: B1 Moderate Competency (Competent Performance)")
                    else:
                        st.error("Target Level: Underachieved Minimum Band Standards")

                # Detailed Metrics Breakdown
                st.markdown("#### Enabling Skills Matrix (Trait System Breakdown)")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric("Content Match", f"{content_score} / 3")
                metric_col2.metric("Form Requirement", f"{form_score} / 2")
                metric_col3.metric("Structure/Coherence", f"{dev_score} / 2")
                metric_col4.metric("Grammar Accuracy", f"{grammar_score} / 2")
                
                metric_col5, metric_col6, metric_col7 = st.columns(3)
                metric_col5.metric("Linguistic Range", f"{linguistic_score} / 2")
                metric_col6.metric("Vocabulary Skill", f"{vocab_score} / 2")
                metric_col7.metric("Spelling Check", f"{spelling_score} / 2")

                # --- NEW: ADVANCED DETAILED ESSAY ANALYSIS PANEL ---
                st.markdown("---")
                st.markdown("## 🔍 Deep Academic Diagnostic Report")
                
                tab1, tab2, tab3 = st.tabs(["🏛️ Paragraph & Architecture", "⚡ Sentence Structure & Range", "📈 Vocabulary & Style"])
                
                with tab1:
                    st.markdown("### Paragraph Architecture Audit")
                    st.write(f"**Total Paragraph Blocks Detected:** {num_paragraphs}")
                    
                    if num_paragraphs == 4:
                        st.success("🎯 **Optimal Structure:** Your essay utilizes the ideal 4-paragraph layout (Introduction, Body Paragraph 1, Body Paragraph 2, Conclusion). This provides maximum structural points in Written Discourse.")
                    elif num_paragraphs == 5:
                        st.info("💡 **5-Paragraph Structure:** Good layout. You have divided your body sections into 3 chunks. Ensure each maintains distinct logical focus.")
                    else:
                        st.warning(f"⚠️ **Structural Warning:** You wrote {num_paragraphs} paragraph block(s). To pass PTE automated grading, use the double-enter space key to divide your text clearly into an Introduction, two focused Body arguments, and a crisp final Conclusion.")
                    
                    st.markdown("#### Paragraph Length Analysis")
                    for idx, para in enumerate(paragraphs):
                        p_words = len(para.split())
                        st.write(f"- **Paragraph {idx+1}:** {p_words} words")
                
                with tab2:
                    st.markdown("### Grammar & Syntactic Complexities")
                    avg_sentence_len = round(word_count / len(sentences)) if sentences else 0
                    st.write(f"**Average Sentence Length:** {avg_sentence_len} words")
                    
                    # Sentence profile breakdown
                    complex_count = len(found_complex)
                    st.write(f"**Subordinate/Complex Joining Clauses Filtered:** {complex_count}")
                    
                    if complex_count >= 3:
                        st.success("✅ **Linguistic Variety:** Great usage of sub-clauses! Mixing structures with transitional words shows high operational linguistic mastery.")
                    else:
                        st.warning("⚠️ **Sentence Variety Improvement:** Your writing relies heavily on simple sentences. Boost your **Linguistic Range** trait marks by inserting sub-clauses via relational markers like *'although'*, *'whereas'*, or *'even though'*.")

                    if grammar_errors:
                        st.markdown("#### Found Grammatical Discrepancies")
                        for err in grammar_errors:
                            st.write(f"❌ {err}")
                    else:
                        st.success("✅ **Sentence Mechanics:** No basic structural capitalization drops tracked.")

                with tab3:
                    st.markdown("### Lexical Variety Profile")
                    st.write(f"**Lexical Diversity Ratio:** {round(lexical_diversity * 100, 1)}% unique terms")
                    
                    # Core academic signpost check
                    st.markdown("#### Academic Signpost Connectors Used")
                    if found_cohesives:
                        st.success(", ".join([f"*{c}*" for c in found_cohesives]))
                    else:
                        st.error("❌ No core academic transitions detected! You must include words like *'However'*, *'Furthermore'*, or *'Therefore'* to establish coherence between ideas.")

                    if spelling_errors:
                        st.markdown("#### 📌 Typos & Repeated Key Flags")
                        st.error(", ".join(spelling_errors))
                        st.caption("Review these flagged items for accidental repeating keys or keyboard typing anomalies.")
                    else:
                        st.success("✅ **Spelling Quality:** No major keyboard mash anomalies detected.")
