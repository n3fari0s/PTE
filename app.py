import streamlit as st
import whisper
import numpy as np
import io
import soundfile as sf
import librosa
from jiwer import wer
from gtts import gTTS
import random

# --- Global App Configurations ---
st.set_page_config(page_title="PTE Academic Speaking Suite", page_icon="🎓", layout="centered")

# --- Shared Global Model Loading (Cached) ---
@st.cache_resource
def load_whisper_shared_engine():
    return whisper.load_model("tiny")

whisper_model = load_whisper_engine = load_whisper_shared_engine()

# --- Helper Function: Text-To-Speech Generation ---
def get_audio_prompt_bytes(text, tld='com'):
    tts = gTTS(text=text, lang='en', tld=tld, slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

# ==========================================
# --- MODULE DATA BANKS & ENGINES ---
# ==========================================

READ_ALOUD_BANK = [
    "Market research is a vital component of any business strategy. It involves gathering information about target markets and customers to understand their needs, buying habits, and preferences.",
    "The development of sustainable energy sources is crucial for reducing carbon emissions. Wind turbines and solar panels are becoming increasingly efficient and cost-effective alternatives to fossil fuels.",
    "Artificial intelligence is rapidly transforming the global landscape. Machine learning algorithms process massive quantities of complex data to uncover hidden patterns, optimize supply chains, and automate manual tasks.",
    "The human brain is a highly complex organ that relies on billions of interconnected neurons. These neural networks communicate through electrochemical signals to process environmental stimuli and regulate physiological systems."
]

REPEAT_SENTENCE_BANK = [
    "Please turn off your mobile phones before entering the lecture theater.",
    "The temporary library closure has been extended until next Monday morning.",
    "The chemical reactions must be monitored closely under constant temperature controls.",
    "An advanced degree in engineering opens up diverse opportunities in global infrastructure development.",
    "The financial report shows a significant increase in profits compared to last quarter.",
    "Relevant data was collected from various reliable academic sources."
]

LECTURE_BANK = [
    {
        "topic": "Membrane Bioreactors (MBR) in Wastewater Treatment",
        "transcript": "Membrane Bioreactors, or MBRs, combine conventional biological treatment, like activated sludge, with membrane filtration. By replacing secondary clarifiers with microfiltration or ultrafiltration membranes, MBR systems achieve superior solid-liquid separation. This high filtration efficiency allows the bioreactor to operate at much higher mixed liquor suspended solids configurations, leading to a smaller footprint and exceptional effluent quality suitable for reclamation.",
        "keywords": ["membrane", "bioreactor", "filtration", "sludge", "separation", "solid", "liquid", "effluent", "wastewater", "treatment"]
    },
    {
        "topic": "The Shift to Renewable Energy Power Grids",
        "transcript": "Transitioning modern electrical networks to renewable infrastructure presents steep engineering hurdles. Unlike steady coal or natural gas thermal generation, solar and wind yields are inherently intermittent due to shifting weather patterns. To stabilize grid frequencies and prevent blackouts, utility operators are deploying massive utility-scale lithium-ion battery storage arrays alongside advanced SCADA automation tools to dynamic-balance load distribution changes.",
        "keywords": ["renewable", "energy", "grid", "intermittent", "solar", "wind", "battery", "storage", "scada", "automation"]
    },
    {
        "topic": "The Mechanics of Macroeconomic Inflation",
        "transcript": "Central banks closely track inflation metrics to gauge baseline economic health. Demand-pull inflation occurs when aggregate consumer demand outpaces aggregate industrial production capacities, effectively pulling prices upward. Conversely, cost-push inflation triggers when spikes in foundational supply inputs, such as crude oil or labor, increase wholesale production costs, forcing firms to transfer those operational expenses directly to buyers.",
        "keywords": ["inflation", "central banks", "demand", "supply", "production", "costs", "economy", "prices"]
    }
]

SITUATION_BANK = [
    {
        "scenario": "You are attending a lecture, but the professor is speaking too quietly, and you cannot hear the points clearly from the back row. You want to ask them to speak louder.",
        "question": "What would you say to the professor?",
        "keywords": ["professor", "speak up", "louder", "hear", "back", "sorry", "excuse me"]
    },
    {
        "scenario": "Your group is working on an engineering project assignment due tomorrow, but one team member has not submitted their section yet and isn't answering texts.",
        "question": "You finally call them and they pick up. What do you say?",
        "keywords": ["project", "assignment", "due", "tomorrow", "section", "update", "status", "help"]
    },
    {
        "scenario": "You borrowed a textbook from your classmate, but accidentally spilled coffee on the cover, staining it badly. You are returning it to them now.",
        "question": "How do you explain this and apologize to your classmate?",
        "keywords": ["sorry", "apologize", "book", "textbook", "spilled", "coffee", "stain", "replace", "buy"]
    }
]

def generate_random_discussion():
    scenarios = [
        {
            "topic": "The value of High-Stakes Final Exams vs. Continuous Assessment.",
            "s1": "Final exams are completely outdated. Last semester, having three finals on the same day caused immense stress, leading me to focus purely on memorizing facts rather than developing a deep understanding of the concepts.",
            "s2": "I completely agree. Standardized tests encourage temporary cramming instead of long-term knowledge retention. I much prefer completing research papers or cumulative projects where we can critically apply what we've learned.",
            "s3": "Actually, final exams serve a clear structural purpose. They force us to comprehensively review the entire semester's material, which creates a helpful sense of academic synthesis, closure, and self-discipline.",
            "keywords": ["exams", "assessment", "cramming", "retention", "projects", "stress"]
        },
        {
            "topic": "Integrating Generative AI tools like ChatGPT into Academic Research.",
            "s1": "Using AI assistants for drafting literature reviews should be accepted. It dramatically speeds up the initial brainstorming process and helps structure scattered analytical thoughts efficiently.",
            "s2": "That is a slippery slope. Over-relying on artificial intelligence fundamentally compromises critical writing skills and increases the risk of accidental plagiarism or hallucinated citations.",
            "s3": "We can't just ban it. Universities should teach students how to treat AI as an interactive research companion, ensuring rigorous fact-checking and ethical disclosure guidelines are maintained.",
            "keywords": ["artificial intelligence", "research", "plagiarism", "writing", "ethics", "brainstorming"]
        }
    ]
    selected = random.choice(scenarios)
    intro_phrases = [
        f"Speaker 1 (Student A): Let's map out our strategy for our upcoming task on {selected['topic'].lower()} ",
        f"Speaker 1 (Group Leader): Thanks for meeting up at the library. We need to finalize our approach regarding {selected['topic'].lower()} "
    ]
    audio_script = (
        f"{random.choice(intro_phrases)}"
        f"{selected['s1']} "
        f"Speaker 2 (Student B): I see exactly where you are coming from, but look at it from this angle. {selected['s2']} "
        f"Speaker 3 (Student C): Let's think about this systematically before we make a final decision. {selected['s3']} "
        f"Speaker 1: That is an excellent point. It seems we have a few different perspectives here, but finding a middle ground will help us move forward effectively."
    )
    return {
        "topic": selected["topic"],
        "audio_script": audio_script,
        "keywords": ["university", "students", "discussion", "assignment", "academic"] + selected["keywords"]
    }

# ==========================================
# --- SIDEBAR NAVIGATION ---
# ==========================================

st.sidebar.title("🎮 Suite Navigation Panel")
module_choice = st.sidebar.selectbox(
    "Select PTE Task Module:",
    [
        "Read Aloud",
        "Repeat Sentence",
        "Retell Lecture",
        "Respond to a Situation",
        "Summarize Group Discussion"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Core System Info:** Running local optimized transcription pipeline.")

# ==========================================
# --- MODULE 1: READ ALOUD ---
# ==========================================
if module_choice == "Read Aloud":
    st.title("🎓 Fast PTE Academic Speaking Engine")
    st.caption("Module 1: Read Aloud • Word Error Rate Analysis")
    
    if "read_aloud_prompt" not in st.session_state:
        st.session_state.read_aloud_prompt = random.choice(READ_ALOUD_BANK)

    target_text = st.session_state.read_aloud_prompt
    st.info(f"**Target Reading Text:** \n\n {target_text}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔊 Listen to Prompt"):
            audio_bytes = get_audio_prompt_bytes(target_text)
            st.audio(audio_bytes, format="audio/mp3")
    with col2:
        if st.button("🔄 Generate Random New Text"):
            st.session_state.read_aloud_prompt = random.choice(READ_ALOUD_BANK)
            st.rerun()

    st.markdown("---")
    audio_value = st.audio_input("Click record and read the text aloud:")

    if audio_value:
        st.info("⚡ Processing audio wave data...")
        data, samplerate = sf.read(io.BytesIO(audio_value.read()))
        if len(data.shape) > 1: data = data[:, 0]
        audio_data = data.astype(np.float32)
        duration_seconds = len(audio_data) / samplerate
        
        with st.spinner("Transcribing..."):
            result = whisper_model.transcribe(audio_data, fp16=False, language="en")
            transcription = result.get("text", "").strip()

        if not transcription:
            st.error("Could not capture any speech. Please check your mic settings.")
        else:
            total_words = len(transcription.split())
            wpm = round((total_words / duration_seconds) * 60) if duration_seconds > 0 else 0
            
            try:
                error_rate = wer(target_text.lower(), transcription.lower())
                content_score = max(10, round(90 * (1 - error_rate)))
            except:
                content_score = 10
                
            if 110 <= wpm <= 170: fluency_score = 85
            elif 90 <= wpm or wpm > 170: fluency_score = 65
            elif wpm > 50: fluency_score = 45
            else: fluency_score = 10

            overall_pte = max(10, min(90, round((content_score + fluency_score) / 2)))
            cefr = "C1/C2 Mastery" if overall_pte >= 76 else "B2" if overall_pte >= 59 else "B1" if overall_pte >= 43 else "A2/A1"

            st.success("✨ Analysis Complete!")
            st.metric(label="OVERALL PTE SPEAKING SCORE", value=f"{overall_pte} / 90", delta=cefr, delta_color="off")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Oral Fluency", f"{fluency_score}")
            c2.metric("Content Accuracy", f"{content_score}")
            c3.metric("Pace Speed", f"{wpm} WPM")
            st.markdown("---")
            st.write("**What the system heard:**")
            st.code(transcription)

# ==========================================
# --- MODULE 2: REPEAT SENTENCE ---
# ==========================================
elif module_choice == "Repeat Sentence":
    st.title("🎧 PTE Repeat Sentence Training Lab")
    st.caption("Module 2: Repeat Sentence • Standard Pearson GSE Mapping")
    
    if "current_repeat_target" not in st.session_state:
        st.session_state.current_repeat_target = random.choice(REPEAT_SENTENCE_BANK)

    st.warning("Instructions: Click below to hear the exam prompt. Listen carefully, then immediately repeat it back.")

    col_action1, col_action2 = st.columns([2, 1])
    with col_action1:
        if st.button("🔊 Generate Natural Exam Audio", use_container_width=True):
            with st.spinner("Synthesizing native English prompt..."):
                st.session_state.prompt_audio_bytes = get_audio_prompt_bytes(st.session_state.current_repeat_target, tld='co.uk')

        if "prompt_audio_bytes" in st.session_state:
            st.audio(st.session_state.prompt_audio_bytes, format="audio/mp3")

    with col_action2:
        if st.button("🔄 Next Sentence", use_container_width=True):
            st.session_state.current_repeat_target = random.choice(REPEAT_SENTENCE_BANK)
            if "prompt_audio_bytes" in st.session_state: del st.session_state.prompt_audio_bytes
            st.rerun()

    if st.checkbox("👁️ Practice Mode: Reveal Target Sentence Text"):
        st.info(f"Target String: **\"{st.session_state.current_repeat_target}\"**")

    st.markdown("---")
    audio_recording = st.audio_input("Record your response:")

    if audio_recording:
        st.info("⚡ Audio captured. Scoring performance profile...")
        data, samplerate = sf.read(io.BytesIO(audio_recording.read()))
        if len(data.shape) > 1: data = data[:, 0]
        audio_data = data.astype(np.float32)
        duration = librosa.get_duration(y=audio_data, sr=samplerate)
        
        with st.spinner("Transcribing your audio input..."):
            result = whisper_model.transcribe(audio_data, fp16=False, language="en")
            transcription = result.get("text", "").strip()
            
        if not transcription:
            st.error("No speech tracked. Please try speaking closer to the microphone.")
        else:
            total_words = len(transcription.split())
            wpm = round((total_words / duration) * 60) if duration > 0 else 0
            
            try:
                word_error = wer(st.session_state.current_repeat_target.lower(), transcription.lower())
                content_score = max(10, round(90 * (1 - word_error)))
            except:
                content_score = 10
                
            if 110 <= wpm <= 170: fluency_score = 88
            elif 85 <= wpm or wpm > 170: fluency_score = 68
            elif wpm > 45: fluency_score = 40
            else: fluency_score = 10
                
            overall_score = max(10, min(90, round((content_score + fluency_score) / 2)))
            
            st.markdown("### 📊 Your Performance Results")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("OVERALL SCORE", f"{overall_score} / 90")
            metric_col2.metric("Content Match", f"{content_score} / 90")
            metric_col3.metric("Fluency Rhythm", f"{fluency_score} / 90")
            
            st.write(f"**Pacing Metrics:** {wpm} Words Per Minute")
            st.markdown("#### Text Comparison")
            st.write("**What you should have said:**")
            st.success(st.session_state.current_repeat_target)
            st.write("**What the engine heard:**")
            st.code(transcription)

# ==========================================
# --- MODULE 3: RETELL LECTURE ---
# ==========================================
elif module_choice == "Retell Lecture":
    st.title("🎙️ PTE Retell Lecture Training Lab")
    st.caption("Module 3: Retell Lecture • Academic Concept Weight Scoring")
    
    if "current_lecture" not in st.session_state:
        st.session_state.current_lecture = random.choice(LECTURE_BANK)

    st.warning("Instructions: Listen below, take quick notes, then record your 40-second summary report.")
    st.info(f"📋 **Current Task Topic:** {st.session_state.current_lecture['topic']}")

    col_act1, col_act2 = st.columns([2, 1])
    with col_act1:
        if st.button("🔊 Play Lecture Audio", use_container_width=True):
            with st.spinner("Streaming lecture narration..."):
                st.session_state.lecture_audio_bytes = get_audio_prompt_bytes(st.session_state.current_lecture['transcript'], tld='co.uk')

        if "lecture_audio_bytes" in st.session_state:
            st.audio(st.session_state.lecture_audio_bytes, format="audio/mp3")

    with col_act2:
        if st.button("🔄 Next Lecture Topic", use_container_width=True):
            st.session_state.current_lecture = random.choice(LECTURE_BANK)
            if "lecture_audio_bytes" in st.session_state: del st.session_state.lecture_audio_bytes
            st.rerun()

    if st.checkbox("👁️ Practice Mode: Reveal Lecture Script Transcript"):
        st.write(st.session_state.current_lecture['transcript'])

    st.markdown("---")
    audio_recording = st.audio_input("Record your summary answer:")

    if audio_recording:
        st.info("⚡ Processing response waveform arrays...")
        data, samplerate = sf.read(io.BytesIO(audio_recording.read()))
        if len(data.shape) > 1: data = data[:, 0]
        audio_data = data.astype(np.float32)
        duration = len(audio_data) / samplerate
        
        with st.spinner("Transcribing performance..."):
            result = whisper_model.transcribe(audio_data, fp16=False, language="en")
            transcription = result.get("text", "").strip()
            
        if not transcription:
            st.error("No speech tracked. Please try again.")
        else:
            total_words = len(transcription.split())
            wpm = round((total_words / duration) * 60) if duration > 0 else 0
            
            matched_keywords = [word for word in st.session_state.current_lecture['keywords'] if word in transcription.lower()]
            kd = len(matched_keywords) / len(st.session_state.current_lecture['keywords']) if st.session_state.current_lecture['keywords'] else 0
            content_score = max(10, min(90, round(10 + (kd * 80))))
            
            if 110 <= wpm <= 160: fluency_score = 88
            elif 80 <= wpm < 110 or 160 < wpm <= 190: fluency_score = 65
            else: fluency_score = 35
                
            overall_score = max(10, min(90, round((content_score + fluency_score) / 2)))
            
            st.success("✨ Evaluation Finalized!")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("OVERALL SCORE", f"{overall_score} / 90")
            m_col2.metric("Content / Retell Accuracy", f"{content_score} / 90")
            m_col3.metric("Oral Fluency", f"{fluency_score} / 90")
            
            st.write(f"**Pacing Metrics:** {wpm} WPM (Response Duration: {duration:.1f}s)")
            st.markdown("#### Transcript Review")
            st.code(transcription)
            
            st.write("🎯 **Core lecture concepts captured successfully:**")
            if matched_keywords: st.success(", ".join(matched_keywords))
            else: st.warning("No high-value technical keywords caught.")

# ==========================================
# --- MODULE 4: RESPOND TO A SITUATION ---
# ==========================================
elif module_choice == "Respond to a Situation":
    st.title("🗣️ PTE Respond to a Situation Lab")
    st.caption("Module 4: Respond to a Situation • Real-world Scenario Logic")
    
    if "current_situation" not in st.session_state:
        st.session_state.current_situation = random.choice(SITUATION_BANK)

    st.warning("Instructions: Listen to the situation context, then record a natural resolution response.")
    st.info(f"📋 **The Situation:**\n\n{st.session_state.current_situation['scenario']}")
    st.markdown(f"**Question:** *{st.session_state.current_situation['question']}*")

    col_act1, col_act2 = st.columns([2, 1])
    with col_act1:
        if st.button("🔊 Play Situation Audio Prompt", use_container_width=True):
            with st.spinner("Generating audio narration..."):
                full_narration = f"{st.session_state.current_situation['scenario']} {st.session_state.current_situation['question']}"
                st.session_state.situation_audio = get_audio_prompt_bytes(full_narration, tld='co.uk')

        if "situation_audio" in st.session_state:
            st.audio(st.session_state.situation_audio, format="audio/mp3")

    with col_act2:
        if st.button("🔄 Next Situation", use_container_width=True):
            st.session_state.current_situation = random.choice(SITUATION_BANK)
            if "situation_audio" in st.session_state: del st.session_state.situation_audio
            st.rerun()

    st.markdown("---")
    audio_recording = st.audio_input("Record your spoken response:")

    if audio_recording:
        st.info("⚡ Processing audio metrics...")
        data, samplerate = sf.read(io.BytesIO(audio_recording.read()))
        if len(data.shape) > 1: data = data[:, 0]
        audio_data = data.astype(np.float32)
        duration = len(audio_data) / samplerate
        
        with st.spinner("Processing speech transcription..."):
            result = whisper_model.transcribe(audio_data, fp16=False, language="en")
            transcription = result.get("text", "").strip()
            
        if not transcription:
            st.error("No audio detected. Please check your mic connection and try again.")
        else:
            total_words = len(transcription.split())
            wpm = round((total_words / duration) * 60) if duration > 0 else 0
            
            matched_keywords = [word for word in st.session_state.current_situation['keywords'] if word in transcription.lower()]
            kd = len(matched_keywords) / len(st.session_state.current_situation['keywords']) if st.session_state.current_situation['keywords'] else 0
            content_score = max(10, min(90, round(10 + (kd * 80))))
            
            if 100 <= wpm <= 160: fluency_score = 88
            elif 70 <= wpm < 100 or 160 < wpm <= 190: fluency_score = 65
            else: fluency_score = 30
                
            overall_score = round((content_score + fluency_score) / 2)
            
            st.success("✨ Score Calculated!")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("OVERALL SCORE", f"{overall_score} / 90")
            m_col2.metric("Appropriateness", f"{content_score} / 90")
            m_col3.metric("Oral Fluency", f"{fluency_score} / 90")
            
            st.write(f"**Pacing Metrics:** {wpm} WPM (Duration: {duration:.1f}s)")
            st.markdown("#### Transcript Review")
            st.code(transcription)
            
            st.write("🎯 **Key contextual terms detected:**")
            if matched_keywords: st.success(", ".join(matched_keywords))
            else: st.warning("No expected situation keywords detected.")

# ==========================================
# --- MODULE 5: SUMMARIZE GROUP DISCUSSION ---
# ==========================================
elif module_choice == "Summarize Group Discussion":
    st.title("👥 PTE Academic Group Discussion Lab")
    st.caption("Module 5: Summarize Group Discussion • Argument Synthesis Logic")
    
    if "discussion_data" not in st.session_state:
        st.session_state.discussion_data = generate_random_discussion()

    st.warning("Instructions: Listen carefully to the academic dispute, take down stances, then record your summary.")
    st.info(f"📋 **Current Academic Exam Topic:**\n\n{st.session_state.discussion_data['topic']}")

    col_act1, col_act2 = st.columns([2, 1])
    with col_act1:
        if st.button("🔊 Synthesize & Play Academic Audio", use_container_width=True):
            with st.spinner("Generating academic discussion voice track..."):
                st.session_state.disc_audio_bytes = get_audio_prompt_bytes(st.session_state.discussion_data['audio_script'], tld='co.uk')

        if "disc_audio_bytes" in st.session_state:
            st.audio(st.session_state.disc_audio_bytes, format="audio/mp3")

    with col_act2:
        if st.button("🔄 Next Exam Prompt", use_container_width=True):
            st.session_state.discussion_data = generate_random_discussion()
            if "disc_audio_bytes" in st.session_state: del st.session_state.disc_audio_bytes  
            st.rerun()

    if st.checkbox("👁️ Reveal Text Script Transcript (Practice Mode)"):
        st.write(st.session_state.discussion_data['audio_script'])

    st.markdown("---")
    audio_recording = st.audio_input("Record your summary synthesis (Target: 90-120 seconds):")

    if audio_recording:
        st.info("⚡ Extracting spoken audio data...")
        data, samplerate = sf.read(io.BytesIO(audio_recording.read()))
        if len(data.shape) > 1: data = data[:, 0]
        audio_data = data.astype(np.float32)
        duration = len(audio_data) / samplerate
        
        with st.spinner("Whisper transcribing performance details..."):
            result = whisper_model.transcribe(audio_data, fp16=False, language="en")
            transcription = result.get("text", "").strip()
            
        if not transcription:
            st.error("No audio content detected. Please check your mic configuration.")
        else:
            total_words = len(transcription.split())
            wpm = round((total_words / duration) * 60) if duration > 0 else 0
            
            matched_keywords = [word for word in st.session_state.discussion_data['keywords'] if word in transcription.lower() and len(word) > 3]
            kd = len(matched_keywords) / len(st.session_state.discussion_data['keywords']) if st.session_state.discussion_data['keywords'] else 0
            content_score = max(10, min(90, round(10 + (kd * 80))))
            
            if 120 <= wpm <= 160: fluency_score = 88
            elif 90 <= wpm < 120 or 160 < wpm <= 190: fluency_score = 68
            else: fluency_score = 38
                
            overall_score = max(10, min(90, round((content_score + fluency_score) / 2)))
            
            st.success("✨ Performance Analysis Finalized!")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("OVERALL TARGET", f"{overall_score} / 90")
            m_col2.metric("Content / Accuracy", f"{content_score} / 90")
            m_col3.metric("Oral Fluency", f"{fluency_score} / 90")
            
            st.write(f"**Pacing Breakdown:** {wpm} WPM (Total Speaking Time: {duration:.1f}s)")
            st.markdown("#### Spoken Transcript Review")
            st.code(transcription)
            
            st.write("🎯 **Core academic parameters captured successfully:**")
            if matched_keywords: st.success(", ".join(matched_keywords))
            else: st.warning("No high-value discussion parameters tracked.")
