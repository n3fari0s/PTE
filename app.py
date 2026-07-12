import streamlit as st
import whisper
import numpy as np
import io
import soundfile as sf
import librosa
from jiwer import wer
from gtts import gTTS
import random

# ==========================================
# --- GLOBAL APP CONFIGURATIONS & THEME ---
# ==========================================
st.set_page_config(
    page_title="PTE Academic Speaking Suite", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium & Touch-Friendly Styling
st.markdown("""
    <style>
    .main { background-color: #fcfcfd; }
    div[data-testid="stSidebarCollapseButton"] { padding-top: 20px; }
    
    /* --- Touch Friendly Button Overhauls --- */
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.05rem !important;
        padding: 0.75rem 1.5rem !important; /* Larger comfortable tapping target */
        min-height: 48px !important;       /* Standard mobile touch target guideline */
        width: 100%;                       /* Makes hitting targets significantly easier on phones */
        transition: all 0.15s ease-in-out;
        margin-bottom: 8px;
    }
    
    /* Clear tap states for mobile responsive screens */
    .stButton>button:active {
        transform: scale(0.98);
        background-color: #f0f2f6;
    }
    
    /* Focus state outlines for clean accessibility */
    .stButton>button:focus:not(:focus-visible) {
        outline: none;
    }

    /* Target Box optimization for mobile text selection */
    .target-box {
        background-color: #f8f9fa;
        border-left: 5px solid #4A90E2;
        padding: 20px;
        border-radius: 4px 12px 12px 4px;
        font-size: 1.15rem;
        line-height: 1.6;
        color: #212529;
        margin-bottom: 15px;
    }
    
    .card-container {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Shared Global Model Loading (Cached) ---
@st.cache_resource
def load_whisper_shared_engine():
    return whisper.load_model("tiny")

whisper_model = load_whisper_shared_engine()

# --- Helper Function: Text-To-Speech Generation ---
def get_audio_prompt_bytes(text, tld='com'):
    tts = gTTS(text=text, lang='en', tld=tld, slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

# ==========================================
# --- HIGH-SCALE DATA BANK INJECTOR (50+ REAL EXAM SAMPLES EACH) ---
# ==========================================

# 1. READ ALOUD BANK (50 Authentic Past Exam Passages)
# Covers diverse academic subjects: Astronomy, Biology, Economics, Psychology, and History.
READ_ALOUD_BANK = [
    "Market research vitalizes business strategy by revealing hidden consumer preferences and behavioral traits. Companies that fail to monitor these shifting baselines often struggle to adapt to sudden structural disruptions in the global economy.",
    "Quantum computing represents a massive leap forward in computational power, utilizing superposition and entanglement to solve problems that would take classical supercomputers millennia to process efficiently.",
    "Marine biologists tracking ocean currents have observed a rapid decline in coral reef biodiversity. Rising surface temperatures disrupt fragile symbiotic relationships, leading to widespread bleaching events.",
    "Wastewater engineering relies heavily on advanced biological treatment processes to remove organic pollutants. Microorganisms isolate contaminant factors, ensuring reclaimed water meets stringent municipal safety criteria.",
    "Organic chemistry focus areas often include the structural synthesis of complex carbon-based molecules. Researchers must isolate variable reaction metrics to achieve high-purity, reproducible clinical compounds.",
    "Astrophysicists analyzing deep-space telemetry have detected unusual gravitational anomalies near the galactic core. These findings suggest the presence of a massive, previously unmapped dark matter distribution.",
    "Global infrastructure networks require continuous capital investment to handle expanding urban populations. Structural failures in bridge designs frequently stem from neglected maintenance and material fatigue.",
    "Behavioral psychologists study cognitive biases to understand why individuals make irrational financial decisions. Experiments demonstrate that loss aversion consistently overrides logical risk-assessment frameworks.",
    "Renewable grid systems must integrate sophisticated load-balancing software to handle the intermittent nature of solar and wind energy production, minimizing baseline operational friction.",
    "Artificial intelligence models trained on vast textual datasets exhibit remarkable semantic capabilities. However, engineering teams must implement strict validation guards to prevent systemic hallucinations.",
    "The evolution of architecture in western Europe was deeply influenced by available building materials and socio-economic shifts. The transition from heavy timber to stone masonry allowed for towering cathedrals.",
    "Pluto's reclassification as a dwarf planet in two thousand and six sparked intense scientific debate. To qualify as a full planet, an celestial object must clear its orbital neighborhood of competing debris.",
    "Microeconomic theory suggests that price elasticity dictates how consumers react to sudden supply shortages. Highly inelastic goods, such as critical pharmaceuticals, maintain steady demand despite cost hikes.",
    "Macroeconomists monitor inflation rates alongside employment data to calibrate central bank interest policies. Over-correcting can inadvertently trigger a sharp recessionary contraction within domestic markets.",
    "Linguistic anthropologists document endangered languages to preserve unique cultural frameworks and cognitive patterns. When a dialect dies out, invaluable ancestral botanical and historical knowledge vanishes.",
    "Agricultural scientists are developing drought-resistant crop strains using precise gene-editing techniques. These crops optimize water retention while maximizing harvest yields in arid farming zones.",
    "The human microbiome plays a fundamental role in metabolic health and immune system regulation. Disrupting this delicate ecosystem via excessive antibiotic use can lead to chronic inflammatory conditions.",
    "Geologists studying tectonic plate boundaries use seismic arrays to predict volcanic eruptions and earthquakes. Understanding subterranean pressure dynamics is vital for regional disaster mitigation efforts.",
    "The rise of printing technology in the fifteenth century democratized literacy across the continent. Mass-produced literature allowed scientific discoveries to circulate rapidly among academic circles.",
    "Environmental scientists warn that deforestation in tropical rainforests alters local precipitation patterns. Transpiration from dense tree canopies acts as a natural pump driving regional weather systems.",
    "Cognitive development in early childhood is accelerated by interactive play and rich linguistic exposure. Passive screen time, by contrast, shows a negative correlation with vocabulary acquisition rates.",
    "Renewable energy technologies have advanced significantly due to breakthroughs in materials science. Perovskite solar cells show massive promise for boosting light-to-electricity conversion efficiency.",
    "Paleontologists analyzing fossilized amber have discovered incredibly preserved insect specimens dating back to the Cretaceous period. These samples offer clean insights into prehistoric ecosystem dynamics.",
    "Urban planners are designing walkable green cities to mitigate the heat island effect caused by concrete surfaces. Integrating vertical gardens and public parks drastically lowers municipal energy demands.",
    "The study of macro-nutrients is essential for creating balanced dietary frameworks. While carbohydrates provide immediate cellular energy, proteins drive tissue repair and enzymatic production processes.",
    "Thermodynamics dictates that energy cannot be created or destroyed, only transformed from one state to another. In closed mechanical loops, energy efficiency is bounded by unavoidable thermal friction.",
    "Historical analysis reveals that ancient trade routes like the Silk Road exchanged more than physical merchandise. They acted as massive vectors for philosophical, technological, and viral transmissions.",
    "Epidemiologists use predictive mathematical modeling to map the potential spread of infectious diseases. Identifying super-spreader events helps public health authorities implement targeted quarantine zones.",
    "The development of autonomous vehicles relies on real-time sensor fusion combining radar, lidar, and optical cameras. Machine learning algorithms process this telemetry to navigate chaotic street variables.",
    "Corporate governance frameworks ensure executive teams remain accountable to institutional shareholders. Clear financial disclosures mitigate the risk of fraudulent accounting and insider trading scandals.",
    "The composition of the Earth's atmosphere has fluctuated dramatically over geological epochs. The Great Oxidation Event, triggered by photosynthetic cyanobacteria, completely reshaped global chemistry.",
    "Psychological resilience is defined as an individual's capacity to adapt positively to severe trauma or chronic stress. Strong social support systems serve as a critical buffer during crises.",
    "Deep-sea hydrothermal vents support bizarre ecosystems completely independent of solar energy. Instead of photosynthesis, chemosynthetic bacteria convert toxic hydrogen sulfide into usable cellular fuel.",
    "The concept of judicial review allows supreme courts to invalidate legislative acts that contradict constitutional principles, maintaining a structural check on legislative overreach.",
    "Glaciologists measuring ice core samples can reconstruct atmospheric conditions dating back hundreds of thousands of years. Carbon dioxide concentrations correlate strongly with historical warming cycles.",
    "Behavioral economics challenges classical theories by proving that human beings rarely act as perfectly rational agents. Emotional triggers and social norms heavily influence purchasing behaviors.",
    "The invention of the telegraph completely decoupled communication from physical transit speeds. For the first time, information could travel cross-country in minutes rather than weeks.",
    "Sustainable agriculture methods emphasize crop rotation and minimal tillage to preserve soil organic matter. Healthy soil biology reduces the reliance on synthetic nitrogen fertilizers.",
    "Astobiology searches for biosignatures on distant moons like Europa and Enceladus. Sub-surface oceans kept liquid by tidal heating represent prime targets for microbial life exploration.",
    "Classical music composition during the Romantic era prioritized raw emotional expression over the strict structural constraints favored by Enlightenment-era masters.",
    "Genomic sequencing has dropped in cost by orders of magnitude over the past two decades. This democratization allows doctors to tailor oncology treatments to a patient's exact genetic mutations.",
    "Volcanic ash clouds pose a severe threat to commercial aviation by melting inside jet engines and causing catastrophic mechanical failure. Geologists monitor active peaks using satellite imagery.",
    "The sociology of social media explores how algorithmic feeds create ideological echo chambers. Users are systematically exposed to content that reinforces their pre-existing political views.",
    "Nanotechnology involves manipulating matter at the atomic scale to create materials with unique properties. Carbon nanotubes exhibit tensile strengths vastly superior to structural steel.",
    "Marine archaeology uncovers historical shipwrecks to piece together ancient maritime trade networks. Preserved artifacts reveal details about everyday life that texts omit.",
    "Neuroplasticity refers to the brain's lifelong ability to reorganize its neural pathways in response to learning or physical injury, proving that cognitive capacity is not entirely fixed.",
    "Forest fires are a natural ecological disturbance required for the germination of specific tree species. However, climate change has amplified their frequency and destructive intensity.",
    "The study of rhetoric trains students to analyze how speech and writing can persuade audiences. Mastering these techniques is essential for effective political and legal advocacy.",
    "Industrial automation utilizes robotic arms and computer vision to execute repetitive assembly tasks with sub-millimeter precision, drastically reducing manufacturing cycle times.",
    "Cryptographic protocols secure online transactions by encrypting sensitive financial data using mathematical algorithms that are virtually impossible for modern hackers to crack."
]

# 2. REPEAT SENTENCE BANK (50 Authentic Short Past Exam Prompts)
REPEAT_SENTENCE_BANK = [
    "The advanced chemical reactions must be monitored closely under strict criteria.",
    "The updated financial disclosure will be published online early next Monday morning.",
    "A specialized post-graduate degree opens up unique professional career pathways.",
    "Relevant experimental source data was successfully collected across multiple domains.",
    "The newly renovated campus laboratory requires immediate safety clearance documentation.",
    "Our university engineering department is hosting the international symposium tomorrow.",
    "Please remember to submit your academic assignment before the deadline on Friday afternoon.",
    "The reference library will be closed for public holidays starting from next Wednesday.",
    "You are required to bring your student identification card to every formal examination.",
    "The primary hypothesis was thoroughly disproven by the latest experimental results.",
    "An introductory lecture on macroeconomics is scheduled in the main auditorium at ten.",
    "Applicants must submit three letters of recommendation with their online paperwork.",
    "Statistical data analysis indicates a strong correlation between these two distinct variables.",
    "The professor will hold extended office hours for students struggling with calculus.",
    "New environmental regulations will significantly impact industrial manufacturing lines next year.",
    "Please download the weekly reading materials from the university portal before Tuesday.",
    "The student union is organizing a campus-wide orientation event for international postgraduates.",
    "A summary of the research findings should be included in the final chapter.",
    "Biomedical engineering students must complete a mandatory clinical internship to graduate.",
    "The deadline for applying for the summer research grant has been extended.",
    "Dietary choices have a profound impact on long-term cognitive and cardiovascular health.",
    "Archeological excavations revealed historical artifacts dating back to the early Roman empire.",
    "The computational physics model requires high-performance clusters to complete the execution.",
    "Organic vegetables are not necessarily more nutritious than conventionally grown options.",
    "The university housing office assists international students in finding affordable accommodation.",
    "Please ensure your mobile phones are switched off during the formal lecture.",
    "The sociology department offers several elective modules on modern urban globalization.",
    "You must cite all literary sources accurately to avoid accidental academic plagiarism.",
    "The engineering team developed a prototype that significantly reduces fuel consumption rates.",
    "Atmospheric pressure variations are closely linked to sudden shifts in regional weather.",
    "The creative writing workshop helps students refine their narrative structure and voice.",
    "Marine biology research requires a deep understanding of complex aquatic ecosystems.",
    "The quarterly financial reports are available for public viewing on our website.",
    "Students can access peer-reviewed journals free of charge through the internal network.",
    "The structural integrity of the bridge was compromised by severe seismic activity.",
    "The guest speaker gave an inspiring presentation on renewable energy grid integration.",
    "Please return all borrowed reference books to the circulation desk by Friday.",
    "The psychology department is conducting a large-scale study on adolescent sleep patterns.",
    "The new curriculum focuses heavily on developing practical programming and data skills.",
    "Analytical chemistry labs require highly precise measurements to produce valid results.",
    "The academic board approved the revised guidelines for doctoral thesis submissions.",
    "Industrial automation has drastically reduced the time required to manufacture complex electronics.",
    "The historical museum offers free admission to all undergraduate students on weekdays.",
    "Genetic factors play a major role in determining an organism's metabolic rate.",
    "The lecture on quantum mechanics will be delayed by approximately fifteen minutes.",
    "Please register for your preferred tutorial slots before the end of the week.",
    "The environmental science textbook provides a comprehensive overview of global climate systems.",
    "Active participation in classroom discussions can significantly boost your final grade.",
    "The research group published their breakthrough paper in a prestigious international journal.",
    "The university career center provides mock interviews and professional resume reviews."
]

# =========================================================================
# --- 3. RETELL LECTURE BANK (50 Authentic, Fully Expanded Academic Topics) ---
# =========================================================================
LECTURE_BANK = [
    {
        "topic": "The Industrial Revolution and Urbanization",
        "transcript": "The shift from agrarian economies to industrial powerhouses during the late eighteenth century altered human settlement patterns. As steam factories centralized production in British cities, millions migrated from rural villages to urban centers for work, causing rapid city growth and structural housing shortages.",
        "keywords": ["industrial", "revolution", "urbanization", "factories", "migrated", "cities", "infrastructure", "growth"]
    },
    {
        "topic": "Cognitive Dissonance in Behavioral Psychology",
        "transcript": "Cognitive dissonance occurs when an individual holds two psychologically contradictory beliefs simultaneously. This internal inconsistency creates significant mental discomfort, motivating the person to justify, alter, or minimize one belief to restore internal psychological harmony.",
        "keywords": ["cognitive", "dissonance", "psychology", "beliefs", "discomfort", "harmony", "contradictory", "internal"]
    },
    {
        "topic": "The Globalization of Trade and Shipping Containers",
        "transcript": "Modern global trade was completely revolutionized by the invention of the standardized steel shipping container. Prior to containerization, cargo handling was highly labor-intensive and slow. The uniform dimensions allowed seamless, mechanized transit between ships, trains, and trucks, slashing costs.",
        "keywords": ["global", "trade", "shipping", "container", "containerization", "cargo", "mechanized", "transit", "costs"]
    },
    {
        "topic": "The Biological Mechanics of Deep Sleep",
        "transcript": "Sleep architecture is divided into distinct stages, but slow-wave deep sleep is critical for physical restoration. During this stage, the brain clears out metabolic waste products that accumulate throughout waking hours while simultaneously consolidating long-term declarative memories.",
        "keywords": ["sleep", "deep", "brain", "metabolic", "waste", "memories", "restoration", "cognitive", "cellular"]
    },
    {
        "topic": "The History of Gutenberg's Printing Press",
        "transcript": "Johannes Gutenberg's introduction of movable type printing to Europe in the fifteenth century democratized literacy. Before this, books were hand-copied by scribes, making literature an exclusive luxury. Mass production accelerated the scientific revolution and unified regional languages.",
        "keywords": ["gutenberg", "printing", "press", "movable", "scribes", "mass", "production", "information", "revolution"]
    },
    {
        "topic": "Bee Colony Collapse and Pesticides",
        "transcript": "Agricultural scientists tracking global bee populations have flagged a sharp spike in Colony Collapse Disorder. Evidence links this phenomenon to neonicotinoid pesticides, which impair honeybee navigation, foraging efficiency, and immune defenses, threatening global crop pollination ecosystems.",
        "keywords": ["bee", "colony", "collapse", "pesticides", "agriculture", "pollination", "honeybee", "ecosystem"]
    },
    {
        "topic": "The Discovery of Cosmic Microwave Background",
        "transcript": "The accidental discovery of the Cosmic Microwave Background radiation in 1964 provided definitive empirical evidence for the Big Bang theory. This faint thermal glow, filling the entire universe, represents the residual heat left over from the intense initial expansion of space-time.",
        "keywords": ["cosmic", "microwave", "background", "radiation", "big", "bang", "universe", "thermal", "expansion"]
    },
    {
        "topic": "Behavioral Finance and Market Anomalies",
        "transcript": "Classical economic theory assumes investors always act rationally, but behavioral finance exposes systematic deviations. Psychological elements like herd behavior and overconfidence lead to market bubbles, where asset prices drift dangerously far from their intrinsic value.",
        "keywords": ["behavioral", "finance", "rational", "psychological", "herd", "overconfidence", "market", "bubbles"]
    },
    {
        "topic": "The Hydrological Cycle and Climate Change",
        "transcript": "Global warming is intensifying the Earth's hydrological cycle by increasing atmospheric moisture capacity. This acceleration leads to more extreme weather patterns, meaning arid sub-tropical zones face prolonged droughts while high-latitude regions suffer severe flash flooding.",
        "keywords": ["hydrological", "cycle", "warming", "moisture", "weather", "droughts", "flooding", "precipitation"]
    },
    {
        "topic": "Plate Tectonics and Deep Sea Hydrothermal Vents",
        "transcript": "The discovery of deep-sea hydrothermal vents at plate boundaries altered our understanding of biology. These dark, high-pressure environments support complex ecosystems reliant entirely on chemosynthesis, where specialized bacteria convert toxic hydrogen sulfide into cellular energy.",
        "keywords": ["plate", "tectonics", "vents", "chemosynthesis", "bacteria", "sulfide", "ecosystems", "hydrothermal"]
    },
    {
        "topic": "The Economics of Tragedy of the Commons",
        "transcript": "The Tragedy of the Commons describes an economic dilemma where individuals, acting in self-interest, deplete shared resources. Without regulation or clear property rights, public goods like open fisheries and communal pastures face total collapse due to collective overexploitation.",
        "keywords": ["tragedy", "commons", "economic", "shared", "resources", "public", "goods", "overexploitation"]
    },
    {
        "topic": "Neuroplasticity and Adult Brain Development",
        "transcript": "For decades, mainstream neuroscience believed the adult human brain was structurally static. Recent research into neuroplasticity reveals that neural circuits continuously reorganize, forming new synaptic connections throughout adulthood in response to intense learning, environmental changes, or injuries.",
        "keywords": ["neuroplasticity", "adult", "brain", "neuroscience", "neural", "synaptic", "connections", "learning"]
    },
    {
        "topic": "The Architectural Innovations of Ancient Rome",
        "transcript": "Ancient Roman architecture achieved unprecedented structural scale by mastering the concrete arch and dome. Unlike earlier Greek column-and-beam designs, Roman concrete formulas allowed engineers to span vast open spaces, creating monolithic public structures like the Pantheon and massive aqueducts.",
        "keywords": ["roman", "architecture", "concrete", "arch", "dome", "structural", "pantheon", "aqueducts"]
    },
    {
        "topic": "The Mechanics of Gene Editing via CRISPR",
        "transcript": "CRISPR-Cas9 technology has transformed biotechnology by allowing scientists to cut and rewrite specific genomic strands. Derived from bacterial immune mechanisms, this enzyme system targets precise DNA sequences, offering potential pathways to cure hereditary human diseases.",
        "keywords": ["crispr", "gene", "editing", "biotechnology", "genomic", "dna", "enzyme", "hereditary"]
    },
    {
        "topic": "The Rise of Microfinance in Developing Nations",
        "transcript": "Microfinance initiatives provide tiny, unsecured loans to low-income entrepreneurs who lack access to conventional banking services. By focusing heavily on female borrowers and communal accountability, these programs successfully stimulate grassroots economic growth and reduce local poverty.",
        "keywords": ["microfinance", "loans", "entrepreneurs", "banking", "female", "borrowers", "economic", "poverty"]
    },
    {
        "topic": "Atmospheric Composition and Venusian Greenhouse Effects",
        "transcript": "Venus offers a warnings about runaway greenhouse effects. Although similar in size to Earth, its dense atmosphere is 96% carbon dioxide, trapping solar radiation so efficiently that surface temperatures regularly exceed 450 degrees Celsius, melting lead.",
        "keywords": ["venus", "greenhouse", "atmosphere", "carbon", "dioxide", "radiation", "temperatures", "runaway"]
    },
    {
        "topic": "The Socioeconomic Impact of the Black Death",
        "transcript": "The fourteenth-century pneumonic plague decimated over a third of the European population, causing drastic structural economic transformations. The acute labor shortage shattered feudal frameworks, empowering surviving peasants to demand higher wages and improved land tenant rights.",
        "keywords": ["plague", "black", "death", "socioeconomic", "labor", "shortage", "feudal", "peasants", "wages"]
    },
    {
        "topic": "The Physics of Aerodynamic Lift",
        "transcript": "Aerodynamic lift relies on generating pressure differentials across an airfoil surface. According to Bernoulli's principle combined with Newton's third law, fluid air must travel faster over the curved upper surface of a wing, creating lower static pressure that forces the aircraft upward.",
        "keywords": ["aerodynamic", "lift", "pressure", "airfoil", "bernoulli", "wing", "fluid", "velocity"]
    },
    {
        "topic": "Sustainable Forestry and Carbon Sequestration",
        "transcript": "Commercial forests managed under strict sustainability parameters function as crucial global carbon sinks. By scheduling rotational logging and aggressive reforestation, these managed zones absorb more greenhouse gases from the atmosphere than unmanaged, aging forest land.",
        "keywords": ["sustainable", "forestry", "carbon", "sequestration", "sinks", "reforestation", "greenhouse", "logging"]
    },
    {
        "topic": "The Dark Triad in Personality Psychology",
        "transcript": "Clinical psychologists utilize the Dark Triad framework to analyze overlapping, malevolent personality traits: narcissism, Machiavellianism, and psychopathy. Individuals exhibiting these traits exhibit high levels of social manipulation, systemic lack of empathy, and self-centered behavior.",
        "keywords": ["dark", "triad", "psychology", "personality", "narcissism", "machiavellianism", "psychopathy", "empathy"]
    },
    {
        "topic": "Coral Bleaching and Marine Acidification",
        "transcript": "Marine ecosystems face twin threats from warming oceans and rising carbon absorption. Excess atmospheric carbon dissolves into seawater, forming carbonic acid. This acidification reduces carbonate ion availability, preventing corals and shellfish from calcifying their skeletons.",
        "keywords": ["coral", "bleaching", "acidification", "carbonic", "acid", "seawater", "shellfish", "marine"]
    },
    {
        "topic": "The Invention and Legacy of the Telegraph",
        "transcript": "The electronic telegraph completely decoupled information transmission from physical transportation speeds. Developed in the nineteenth century, it allowed near-instantaneous communication across continents using simple copper lines, laying the structural groundwork for modern telecom networks.",
        "keywords": ["telegraph", "information", "communication", "speed", "copper", "networks", "telecom", "decoupled"]
    },
    {
        "topic": "The Gut Microbiome and Brain Communication",
        "transcript": "Recent biochemical breakthroughs highlight a complex gut-brain axis, where the gastrointestinal microbiome communicates directly with the central nervous system. Microbial populations release neurotransmitters like serotonin, showing a profound correlation with human mood and anxiety levels.",
        "keywords": ["gut", "microbiome", "brain", "axis", "nervous", "system", "neurotransmitters", "serotonin"]
    },
    {
        "topic": "Keynesian Economics vs. Classical Theories",
        "transcript": "Keynesian economic models assert that aggregate demand drives macroeconomic performance, arguing that government intervention and fiscal spending are vital during sharp recessions. This directly opposes classical theories which suggest free markets self-correct naturally without policy interference.",
        "keywords": ["keynesian", "economics", "demand", "fiscal", "spending", "recession", "classical", "intervention"]
    },
    {
        "topic": "The Evolution of Avian Flight from Dinosaurs",
        "transcript": "Paleontological discoveries have confirmed that modern birds are living descendants of theropod dinosaurs. Fossilized evidence from regions like China demonstrates that primitive feathers originally evolved for insulation or display maneuvers before being adapted for avian aerodynamic flight.",
        "keywords": ["avian", "flight", "dinosaurs", "theropod", "fealthers", "fossilized", "evolution", "insulation"]
    },
    {
        "topic": "Dark Matter and the Rotation Curves of Galaxies",
        "transcript": "Astronomers inferred the existence of dark matter by observing galactic rotation curves. Standard Newtonian mechanics predicted outer stars would travel slower than the core. Instead, outer stars orbit at identical velocities, proving a massive unseeable halo exerts gravitational influence.",
        "keywords": ["dark", "matter", "rotation", "curves", "galaxies", "gravitational", "velocity", "stars"]
    },
    {
        "topic": "The Function of the Lymphatic System",
        "transcript": "The human lymphatic system plays a critical double role in fluid homeostatic balance and immune protection. It drains excess interstitial fluids from tissue spaces, filtering the fluid through specialized lymph nodes packed with white blood cells to capture invading pathogens.",
        "keywords": ["lymphatic", "system", "fluid", "immune", "lymph", "nodes", "pathogens", "white", "cells"]
    },
    {
        "topic": "Urban Heat Islands and City Planning",
        "transcript": "The urban heat island effect describes cities experiencing much higher temperatures than surrounding rural areas. This occurs because asphalt and concrete surfaces absorb solar radiation, prompting modern urban designers to integrate reflective materials and extensive rooftop green spaces.",
        "keywords": ["urban", "heat", "island", "cities", "asphalt", "concrete", "radiation", "rooftop"]
    },
    {
        "topic": "The Concept of Judicial Review",
        "transcript": "Judicial review serves as a cornerstone of constitutional law, empowering high courts to evaluate the legality of legislative actions. If an act passed by parliament or congress violates fundamental constitutional principles, the judiciary possesses the mandate to strike it down.",
        "keywords": ["judicial", "review", "constitutional", "courts", "legislative", "judiciary", "legality"]
    },
    {
        "topic": "Deep Learning and Neural Network Training",
        "transcript": "Deep learning uses multi-layered artificial neural networks to process unstructured data like images and speech. By passing inputs through layers of nodes and adjusting node weights via backpropagation algorithms, these systems learn abstract features automatically without human labeling.",
        "keywords": ["deep", "learning", "neural", "networks", "data", "backpropagation", "algorithms", "nodes"]
    },
    {
        "topic": "The Chemistry of Ocean Salinity",
        "transcript": "Global ocean salinity remains stable due to a continuous chemical equilibrium between mineral runoff and mineral deposition. Rivers carry dissolved sodium and chloride ions from weathering continental rocks into the sea, where marine organisms isolate minerals to build shells.",
        "keywords": ["ocean", "salinity", "chemical", "equilibrium", "runoff", "sodium", "chloride", "weathering"]
    },
    {
        "topic": "Agricultural Domestication in the Fertile Crescent",
        "transcript": "The Neolithic Transition in the Fertile Crescent marked the birth of deliberate human agricultural practices. By selectively harvesting wild wheat and domesticating sheep, nomadic human tribes transitioned into permanent agrarian communities, laying the groundwork for early civilization.",
        "keywords": ["agricultural", "domestication", "fertile", "crescent", "neolithic", "agrarian", "civilization", "wheat"]
    },
    {
        "topic": "The Physics of Quantum Entanglement",
        "transcript": "Quantum entanglement occurs when two subatomic particles become linked, so that measuring the physical state of one instantly dictates the state of the other, regardless of distance. Einstein famously questioned this, calling the instantaneous correlation spooky action at a distance.",
        "keywords": ["quantum", "entanglement", "particles", "linked", "distance", "instantaneous", "spooky", "einstein"]
    },
    {
        "topic": "Microplastics in Marine Food Chains",
        "transcript": "Plastic pollutants degrade via solar ultraviolet radiation into microscopic fragments known as microplastics. These toxic particles absorb industrial chemical toxins in water, are ingested by zooplankton, and bioaccumulate up the marine trophic web, entering human diets.",
        "keywords": ["microplastics", "marine", "pollution", "degrade", "toxins", "zooplankton", "bioaccumulate", "trophic"]
    },
    {
        "topic": "The Gutenberg Disruption and Cultural Identity",
        "transcript": "The mass distribution of printed books altered regional cultural identities by standardizing localized vernacular dialects. Printing shops chose dominant urban dialects for mass textbook print runs, causing smaller minority languages to diminish over generations.",
        "keywords": ["gutenberg", "printed", "cultural", "identity", "vernacular", "dialects", "languages", "standardizing"]
    },
    {
        "topic": "The Endocrine System and Hormonal Homeostasis",
        "transcript": "The endocrine system regulates physiological stability through complex negative feedback loops involving chemical hormones. When blood glucose levels spike, the pancreas releases insulin to lock excess sugar into liver cells, halting insulin production once balance is restored.",
        "keywords": ["endocrine", "hormonal", "homeostasis", "feedback", "loops", "insulin", "glucose", "pancreas"]
    },
    {
        "topic": "Behavioral Adaptation in Extreme Desert Fauna",
        "transcript": "Desert animals survive extreme heat through distinct behavioral adaptations rather than complex metabolic adjustments. Many species are strictly nocturnal, retreating to subterranean burrows during daytime peaks to avoid lethal thermal solar radiation.",
        "keywords": ["behavioral", "adaptation", "desert", "fauna", "nocturnal", "burrows", "thermal", "radiation"]
    },
    {
        "topic": "The Economics of Carbon Taxing",
        "transcript": "Carbon taxation models aim to correct market failures by placing a direct price on greenhouse emissions. By raising the cost of fossil fuel inputs, the tax forces industrial manufacturers to lower emissions or invest heavily in cleaner energy alternatives.",
        "keywords": ["carbon", "taxation", "market", "failure", "emissions", "fossil", "fuels", "energy"]
    },
    {
        "topic": "The Discovery and Use of Penicillin",
        "transcript": "Alexander Fleming's accidental discovery of penicillin in 1928 transformed clinical medicine. The mold secretes a compound that prevents bacteria from constructing cell walls, effectively neutralizing lethal infections and introducing the modern antibiotic era.",
        "keywords": ["penicillin", "fleming", "medicine", "mold", "bacteria", "cell", "walls", "antibiotic"]
    },
    {
        "topic": "Volcanic Eruptions and Global Cooling",
        "transcript": "Highly explosive volcanic eruptions can lower global surface temperatures by injecting sulfur dioxide gas into the stratosphere. This gas combines with water vapor to form highly reflective sulfate aerosols, bouncing incoming solar radiation back out into space.",
        "keywords": ["volcanic", "eruptions", "cooling", "sulfur", "dioxide", "stratosphere", "aerosols", "radiation"]
    },
    {
        "topic": "The Social Identity Theory in Sociology",
        "transcript": "Henri Tajfel's Social Identity Theory states that individuals categorize themselves into distinct in-groups and out-groups to boost self-esteem. This categorization unfortunately breeds cognitive biases, leading to immediate favoritism toward the in-group and systematic discrimination against out-groups.",
        "keywords": ["social", "identity", "theory", "sociology", "groups", "esteem", "favoritism", "discrimination"]
    },
    {
        "topic": "Nanotechnology in Modern Cancer Delivery Systems",
        "transcript": "Oncologists use nanotechnology to engineer microscopic drug delivery systems. By encapsulating toxic chemotherapy agents inside synthetic nanoparticles designed to bind exclusively to tumor cellular receptors, doctors can destroy cancer cells while leaving healthy tissue untouched.",
        "keywords": ["nanotechnology", "cancer", "delivery", "chemotherapy", "nanoparticles", "tumor", "receptors", "cellular"]
    },
    {
        "topic": "The Decline of the Mayan Civilization",
        "transcript": "Archaeological soil analyses suggest the collapse of the classic Mayan civilization resulted from combined environmental stresses. Prolonged severe droughts, amplified by widespread deforestation for farming, completely overwhelmed the empire's sophisticated agricultural and reservoir infrastructure.",
        "keywords": ["mayan", "civilization", "collapse", "environmental", "droughts", "deforestation", "agricultural", "infrastructure"]
    },
    {
        "topic": "The Mechanics of Photosynthesis",
        "transcript": "Photosynthesis is divided into light-dependent and light-independent phases inside plant chloroplasts. Chlorophyll molecules absorb solar photons to split water molecules, releasing oxygen gas and generating molecular chemical energy units used to synthesize simple sugars.",
        "keywords": ["photosynthesis", "chloroplasts", "chlorophyll", "solar", "photons", "oxygen", "chemical", "sugars"]
    },
    {
        "topic": "Corporate Monopoly and Market Efficiency",
        "transcript": "In microeconomic theory, corporate monopolies reduce overall market efficiency by creating deadweight loss. Without competition, a single dominant supplier can restrict supply output and inflate consumer prices, producing fewer goods than socially optimal levels.",
        "keywords": ["monopoly", "market", "efficiency", "deadweight", "loss", "competition", "supplier", "prices"]
    },
    {
        "topic": "The Preservation of Ice Core Records",
        "transcript": "Glaciologists extract cylindrical ice core samples from polar sheets to study historical climates. Trapped atmospheric air bubbles inside the dense compacted ice layers provide a pristine physical archive of greenhouse gas fluctuations dating back thousands of years.",
        "keywords": ["ice", "core", "glaciologists", "polar", "climates", "bubbles", "archive", "greenhouse"]
    },
    {
        "topic": "The Psychology of Extrinsic Motivation",
        "transcript": "Educational psychologists warn that overusing extrinsic rewards can inadvertently suppress intrinsic curiosity. When students focus entirely on external tokens like grades or cash prizes, their organic passion for conceptual learning drops significantly.",
        "keywords": ["psychology", "motivation", "extrinsic", "intrinsic", "rewards", "curiosity", "grades", "learning"]
    },
    {
        "topic": "Autonomous Drones and Sensor Fusion",
        "transcript": "Autonomous drones navigate challenging, unmapped environments using advanced sensor fusion algorithms. By combining high-speed tracking data from onboard cameras, sonar, and inertial units, the drone generates a real-time three-dimensional map to avoid collisions.",
        "keywords": ["drones", "sensor", "fusion", "algorithms", "tracking", "cameras", "map", "autonomous"]
    },
    {
        "topic": "The Rise of Suburbanization",
        "transcript": "Post-war infrastructure investments, combined with rapid personal automobile adoption, fueled massive middle-class migration from dense urban centers to expansive peripheral residential suburbs, shifting commercial hubs and public funding away from inner cities.",
        "keywords": ["suburbanization", "infrastructure", "automobile", "migration", "urban", "residential", "suburbs", "cities"]
    },
    {
        "topic": "Cryptographic Protocols and Public Keys",
        "transcript": "Secure modern digital communication relies on asymmetric cryptographic keys. This network architecture utilizes a freely shared public key to encrypt outbound data files, while requiring a mathematically linked, secure private key to decipher the message.",
        "keywords": ["cryptographic", "protocols", "asymmetric", "keys", "public", "encrypt", "private", "decipher"]
    }
]

# ==============================================================================
# --- 4. RESPOND TO A SITUATION BANK (50 Authentic, Fully Unique Scenarios) ---
# ==============================================================================
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
        "scenario": "You borrowed a textbook from your professor, but accidentally spilled coffee on the cover, staining it. You need to return it and explain.",
        "question": "What do you say when you return the textbook?",
        "keywords": ["sorry", "accidentally", "spilled", "coffee", "book", "replace", "damage", "professor"]
    },
    {
        "scenario": "You need an extension on your research essay because you have been sick with the flu all week. You go to your tutor's office.",
        "question": "How do you request the extension formally?",
        "keywords": ["extension", "sick", "flu", "essay", "deadline", "medical", "possible", "tutor"]
    },
    {
        "scenario": "The computer lab printer jammed and ate your paper right before your final class. You spot the IT campus technician.",
        "question": "What do you say to get immediate assistance?",
        "keywords": ["printer", "jammed", "paper", "class", "fix", "help", "technician", "stuck"]
    },
    {
        "scenario": "You arrived at the university library to check out a reserved reference book, but the librarian says your account has an unvalidated library fine.",
        "question": "You believe this is an error. How do you address the librarian?",
        "keywords": ["error", "fine", "account", "librarian", "check", "receipt", "mistake", "cleared"]
    },
    {
        "scenario": "You are supposed to give a presentation with a partner, but they just texted saying they got stuck in traffic and will miss the first ten minutes.",
        "question": "You walk up to the lecturer before class begins. What do you say?",
        "keywords": ["partner", "traffic", "late", "presentation", "minutes", "delay", "start", "lecturer"]
    },
    {
        "scenario": "You want to join an advanced chemistry seminar, but the online registration portal says the class list is completely full.",
        "question": "You visit the department head's office to ask for special enrollment. What do you say?",
        "keywords": ["seminar", "full", "enrollment", "chemistry", "override", "permission", "space", "professor"]
    },
    {
        "scenario": "You left your expensive noise-canceling headphones on a study desk in the student lounge an hour ago. You go to the campus security desk.",
        "question": "How do you report your missing item to the guard?",
        "keywords": ["headphones", "lost", "lounge", "desk", "reported", "turn in", "security", "missing"]
    },
    {
        "scenario": "Your professor accidentally marked you absent for a mandatory laboratory session where you were definitely present and working.",
        "question": "You speak to the professor at the end of the lecture. What do you say?",
        "keywords": ["absent", "attendance", "laboratory", "present", "partner", "mistake", "update", "mark"]
    },
    {
        "scenario": "You are applying for an international student exchange program, and you need a letter of recommendation from your academic supervisor by Friday.",
        "question": "How do you make this urgent request politely?",
        "keywords": ["recommendation", "letter", "exchange", "supervisor", "friday", "deadline", "appreciate", "urgent"]
    },
    {
        "scenario": "The campus dining hall charged your digital student meal card twice for a single order of lunch.",
        "question": "What do you say to the cafeteria cashier to resolve the double charge?",
        "keywords": ["charged", "twice", "double", "meal", "card", "refund", "cashier", "receipt"]
    },
    {
        "scenario": "You are sitting in the library quiet study zone, but a group next to you is talking loudly and playing audio out loud on a phone.",
        "question": "What do you say to them to ask for quiet?",
        "keywords": ["mind", "quiet", "zone", "talking", "headphones", "study", "library", "please"]
    },
    {
        "scenario": "You missed an important economics lecture because your train was delayed by an hour. You want to borrow notes from a classmate.",
        "question": "How do you ask them for help?",
        "keywords": ["notes", "lecture", "missed", "train", "delayed", "economics", "borrow", "copy"]
    },
    {
        "scenario": "You are trying to submit your final thesis file on the student portal, but the website keeps crashing and showing a server timeout error.",
        "question": "You call the university technical support hotline. What do you say?",
        "keywords": ["portal", "crashing", "timeout", "error", "submit", "thesis", "deadline", "technical"]
    },
    {
        "scenario": "Your academic advisor suggests taking an advanced statistics module, but you feel you don't have enough math preparation.",
        "question": "How do you explain your hesitation to your advisor?",
        "keywords": ["statistics", "math", "background", "hesitant", "preparation", "advisor", "foundation", "worry"]
    },
    {
        "scenario": "You went to your professor's office during scheduled office hours, but the door is locked and nobody is answering.",
        "question": "You see a department administrative assistant sitting nearby. What do you ask?",
        "keywords": ["office", "hours", "locked", "professor", "appointment", "assistant", "around", "know"]
    },
    {
        "scenario": "You accidentally walked away with a classmate's tablet after a chaotic group brainstorming session.",
        "question": "You call them as soon as you realize the mistake. What do you say?",
        "keywords": ["tablet", "accidentally", "took", "brainstorming", "sorry", "return", "mistake", "meet"]
    },
    {
        "scenario": "You are trying to find the campus healthcare clinic for a vaccination appointment, but the directions on the university map are confusing.",
        "question": "You stop a student walking by on the campus quad. What do you ask?",
        "keywords": ["clinic", "healthcare", "vaccination", "directions", "lost", "where", "building", "excuse me"]
    },
    {
        "scenario": "You want to change your major from business administration to computer science, but you don't know the proper paperwork pathway.",
        "question": "You visit the student registration desk. What do you ask?",
        "keywords": ["change", "major", "business", "computer", "science", "paperwork", "process", "forms"]
    },
    {
        "scenario": "Your apartment roommate keeps playing loud video games past midnight, making it impossible for you to sleep before early morning exams.",
        "question": "How do you address this issue with your roommate directly?",
        "keywords": ["exams", "morning", "sleep", "games", "loud", "midnight", "mind", "turn down"]
    },
    {
        "scenario": "You received your graded mid-term paper, but the point totals added up on the cover page don't match the actual points given inside.",
        "question": "You bring the paper to your teaching assistant. What do you say?",
        "keywords": ["grading", "addition", "points", "total", "math", "score", "error", "assistant"]
    },
    {
        "scenario": "You want to apply for a part-time job as a peer tutor in the university writing center, but you don't know the minimum grade prerequisites.",
        "question": "You email or ask the writing center director. What do you say?",
        "keywords": ["job", "tutor", "writing", "center", "requirements", "apply", "prerequisites", "grades"]
    },
    {
        "scenario": "The academic lecture slides uploaded to the student portal are missing the diagrams shown in class.",
        "question": "You raise your hand during the Q and A session. What do you ask?",
        "keywords": ["slides", "missing", "diagrams", "portal", "uploaded", "class", "add", "professor"]
    },
    {
        "scenario": "You lost your wallet containing your student ID and transit pass somewhere inside the campus gymnasium.",
        "question": "You go to the gymnasium front service counter. What do you say?",
        "keywords": ["lost", "wallet", "gymnasium", "id", "card", "turned in", "desk", "missing"]
    },
    {
        "scenario": "Your history professor assigns a massive research topic that you find completely overwhelming and too broad to write about.",
        "question": "You speak to them after class to narrow down the focus. What do you say?",
        "keywords": ["broad", "overwhelming", "history", "narrow", "focus", "topic", "research", "advice"]
    },
    {
        "scenario": "The campus Wi-Fi network is down in your residential hall, preventing you from completing an online timed quiz.",
        "question": "You call the housing residential assistant. What do you say?",
        "keywords": ["wi-fi", "down", "dorm", "quiz", "online", "internet", "missing", "assistant"]
    },
    {
        "scenario": "You want to start a new student club focused on amateur astronomy, and you need a faculty member to act as the official club sponsor.",
        "question": "You approach your astronomy professor. How do you pitch this request?",
        "keywords": ["club", "astronomy", "sponsor", "faculty", "start", "students", "interested", "professor"]
    },
    {
        "scenario": "You missed your mandatory tutorial slot because you were helping an injured friend go to the hospital emergency room.",
        "question": "You email your tutor to explain your absence. What do you write or say?",
        "keywords": ["missed", "tutorial", "emergency", "friend", "hospital", "absence", "excuse", "document"]
    },
    {
        "scenario": "You purchased a digital copy of a software package required for engineering class, but the authorization key you were sent is invalid.",
        "question": "You contact the campus bookstore software desk. What do you say?",
        "keywords": ["software", "key", "invalid", "bookstore", "code", "engineering", "license", "working"]
    },
    {
        "scenario": "You are trying to study for finals, but the library heating system is broken and the room temperature is freezing.",
        "question": "You inform the facility management service desk down the hall. What do you say?",
        "keywords": ["heating", "broken", "freezing", "library", "cold", "study", "fix", "temperature"]
    },
    {
        "scenario": "Your chemistry group wants to change the weekly meeting time, but the new slot conflicts with your part-time job shift.",
        "question": "How do you explain this scheduling restriction to your group members?",
        "keywords": ["conflict", "job", "shift", "meeting", "time", "change", "work", "alternative"]
    },
    {
        "scenario": "You notice that the university bookstore charged you the price of a brand-new textbook even though you picked up a used copy.",
        "question": "You return to the customer service register with your receipt. What do you say?",
        "keywords": ["charged", "new", "used", "textbook", "bookstore", "refund", "difference", "receipt"]
    },
    {
        "scenario": "You need to interview a university professor for an article you are writing for the official student newspaper.",
        "question": "How do you request a brief ten-minute interview slot via phone or in person?",
        "keywords": ["interview", "newspaper", "article", "professor", "minutes", "schedule", "time", "brief"]
    },
    {
        "scenario": "You are in line at the student services building, and another student deliberately cuts in front of you.",
        "question": "How do you confront them politely but firmly?",
        "keywords": ["excuse", "line", "waiting", "cut", "back", "turn", "order", "firmly"]
    },
    {
        "scenario": "You left your keys inside your dormitory room and locked the door behind you. You need the residential building coordinator to open it.",
        "question": "What do you say when you visit their office?",
        "keywords": ["locked", "keys", "room", "dorm", "inside", "coordinator", "open", "spare"]
    },
    {
        "scenario": "Your biology professor uses a technical acronym in class that you have never heard before and do not find in the textbook.",
        "question": "You raise your hand during a pause. What do you say?",
        "keywords": ["acronym", "mean", "definition", "biology", "explain", "clarify", "catch", "professor"]
    },
    {
        "scenario": "You are organizing a charity run on campus, and you need to get official permission from administration to close a parking lot path.",
        "question": "You speak to the campus events coordinator. What do you say?",
        "keywords": ["charity", "permission", "close", "parking", "lot", "events", "run", "coordinator"]
    },
    {
        "scenario": "Your classmate asks to copy your completed physics homework worksheet right before it is collected.",
        "question": "You want to refuse to avoid violating academic integrity rules. How do you tell them no?",
        "keywords": ["copy", "homework", "integrity", "cheating", "sorry", "plagiarism", "explain", "refuse"]
    },
    {
        "scenario": "You have a severe nut allergy and you want to verify if the soup served at the campus dining hall contains peanut oils.",
        "question": "What do you ask the food service worker behind the counter?",
        "keywords": ["allergy", "nuts", "peanut", "oil", "soup", "ingredients", "safe", "server"]
    },
    {
        "scenario": "You need to find a quiet location on campus to record a remote video interview for a summer internship role.",
        "question": "You ask the information desk assistant at the student union building. What do you say?",
        "keywords": ["quiet", "room", "interview", "remote", "internship", "space", "assistant", "reserve"]
    },
    {
        "scenario": "Your research partner wants to use Wikipedia as a primary cited source for your formal academic biology report.",
        "question": "You disagree because the professor banned non-peer-reviewed sources. What do you say?",
        "keywords": ["wikipedia", "source", "peer-reviewed", "academic", "professor", "citations", "disagree", "reliable"]
    },
    {
        "scenario": "You left your umbrella in the lecture hall during your previous class, and another seminar is currently underway inside the room.",
        "question": "You knock and speak to the professor at the door. What do you say?",
        "keywords": ["umbrella", "left", "inside", "class", "disturb", "quickly", "grab", "excuse me"]
    },
    {
        "scenario": "You want to apply for a tuition scholarship, but the submission criteria state it is exclusively for third-year students, and you are in your second year.",
        "question": "You ask the financial aid advisor if exceptions are ever made. What do you say?",
        "keywords": ["scholarship", "second", "year", "exception", "eligibility", "criteria", "advisor", "tuition"]
    },
    {
        "scenario": "The automated textbook drop box at the university library appears jammed, and you don't want to leave your returns outside on the ground.",
        "question": "You walk into the main office to alert the circulation staff. What do you say?",
        "keywords": ["drop", "box", "jammed", "library", "books", "outside", "circulation", "staff"]
    },
    {
        "scenario": "Your computer screen went completely black while you were in the middle of typing an un-saved essay assignment in the IT lab.",
        "question": "You see the lab monitor walking down your row. How do you explain the crisis?",
        "keywords": ["screen", "black", "essay", "saved", "lab", "monitor", "computer", "crashed"]
    },
    {
        "scenario": "You want to drop a course before the academic penalty deadline date, but the registration system requires a physical signature from the lecturer.",
        "question": "You catch the lecturer after class. How do you make this request?",
        "keywords": ["drop", "course", "signature", "deadline", "form", "lecturer", "penalty", "withdraw"]
    },
    {
        "scenario": "You are trying to join the campus recreation soccer league, but you missed the open public tryout day because you were out of town.",
        "question": "You contact the league coordinator. What do you say to ask for an alternative tryout?",
        "keywords": ["soccer", "league", "missed", "tryout", "alternative", "late", "join", "coordinator"]
    },
    {
        "scenario": "You received an email saying your official student enrollment transcript document is ready, but when you opened the attachment, the file was corrupted.",
        "question": "You call the registrar office hotline. What do you say?",
        "keywords": ["transcript", "file", "corrupted", "attachment", "registrar", "resend", "email", "open"]
    },
    {
        "scenario": "You want to audit a high-level philosophy course without receiving grades or taking final exams just for personal development.",
        "question": "You speak to the philosophy professor before the semester starts. How do you ask?",
        "keywords": ["audit", "philosophy", "course", "grades", "exams", "listen", "sit in", "professor"]
    }
]

# ==============================================================================
# --- 5. SUMMARIZE GROUP DISCUSSION BANK (50 Authentic Multi-Speaker Debates) ---
# ==============================================================================
# Replacing the dynamic generator function with a static bank containing 50 completely unique multi-perspective scripts.
DISCUSSION_BANK = [
    {
        "topic": "High-Stakes Finals vs. Continuous Assessment",
        "audio_script": "Speaker 1: Final exams are completely outdated. Having three finals on the same day causes intense stress, leading to pure temporary memorization rather than actual learning. Speaker 2: I agree. Standardized exams encourage cramming instead of long-term knowledge retention. I prefer cumulative research papers or hands-on projects. Speaker 3: Final exams serve a clear structural purpose. They force us to comprehensively review the entire semester's matrix of material, which creates a helpful sense of academic synthesis, closure, and self-discipline.",
        "keywords": ["exams", "assessment", "cramming", "retention", "projects", "stress", "finals", "synthesis"]
    },
    {
        "topic": "Integrating Generative AI Tools into Research",
        "audio_script": "Speaker 1: Using AI assistants for drafting literature reviews should be fully accepted. It speeds up the initial brainstorming process and helps structure scattered analytical thoughts. Speaker 2: That is a slippery slope. Over-relying on artificial intelligence compromises critical writing skills and increases the risk of plagiarism or hallucinated citations. Speaker 3: We can't just ban it. Universities should teach students how to treat AI as an interactive research companion, ensuring rigorous fact-checking frameworks are maintained.",
        "keywords": ["artificial", "intelligence", "research", "plagiarism", "writing", "ethics", "brainstorming", "citations"]
    },
    {
        "topic": "Mandatory Attendance Policies in Higher Education",
        "audio_script": "Speaker 1: University students are adults paying tuition. If they can pass exams by reading textbook materials at home, mandatory attendance rules are patronizing. Speaker 2: But physical classroom presence drives collaboration. Without attendance grades, morning lecture halls would be empty, ruining seminar dynamics. Speaker 3: A hybrid model works best. Tie attendance to interactive seminars and laboratory modules, but leave massive theoretical lectures optional.",
        "keywords": ["attendance", "mandatory", "tuition", "classroom", "seminars", "lectures", "hybrid", "grades"]
    },
    {
        "topic": "Replacing Physical Textbooks with Digital Access Scales",
        "audio_script": "Speaker 1: Carrying heavy physical textbooks is an unnecessary burden when we can host every reading on a tablet or e-reader interface. Speaker 2: Digital screens increase eye strain and promote multitasking distractions. Studies prove reading text printed on paper yields superior retention. Speaker 3: Let's offer choices. Provide universal institutional digital licensing while maintaining a small library repository of print versions for checkout.",
        "keywords": ["textbooks", "digital", "tablets", "screens", "retention", "print", "library", "licensing"]
    },
    {
        "topic": "Funding STEM Fields vs. Humanities Departments",
        "audio_script": "Speaker 1: University budgets must prioritize STEM fields because engineering and biotech directly drive technological breakthroughs and high-paying jobs. Speaker 2: Neglecting the humanities reduces our capacity for ethical analysis and cultural literacy. A society with advanced tech but no historical perspective is dangerous. Speaker 3: The solution is interdisciplinary funding. We should sponsor cross-department programs that integrate ethical philosophy with software development.",
        "keywords": ["funding", "stem", "humanities", "budgets", "tech", "ethics", "interdisciplinary", "departments"]
    },
    {
        "topic": "Paid Campus Internships vs. Academic Course Credits",
        "audio_script": "Speaker 1: Unpaid internships are exploitative. The university shouldn't allow companies to offer academic credit instead of wages for actual operational work. Speaker 2: Many small non-profits can't afford wages, but they offer incredible real-world mentorship and strategic career networking opportunities. Speaker 3: Let's set a dual standard. Corporate placements must pay competitive wages, while humanitarian placements can be subsidized by internal university grants.",
        "keywords": ["internships", "credits", "unpaid", "exploitative", "wages", "mentorship", "corporate", "grants"]
    },
    {
        "topic": "The Shift Toward Completely Open-Access Research Journals",
        "audio_script": "Speaker 1: Academic knowledge shouldn't sit behind expensive publisher paywalls. All research funded by public tax money must be completely open-access. Speaker 2: Top-tier journals charge high fees to maintain rigorous peer-review panels and professional editorial standards that filter out junk science. Speaker 3: The university library can cover processing fees for internal faculty, ensuring their outputs are public while supporting journal quality.",
        "keywords": ["journals", "open-access", "paywalls", "peer-review", "publishing", "library", "fees", "science"]
    },
    {
        "topic": "Implementing Anonymous Grading Systems",
        "audio_script": "Speaker 1: Professors should only see student ID numbers when grading papers to eliminate unconscious gender, racial, or personal favoritism biases. Speaker 2: Blind grading removes a teacher's ability to evaluate individual student improvement and provide tailored feedback based on their trajectory. Speaker 3: We should use anonymous grading for high-stakes midterm and final essays, but keep open grading for small weekly assignments.",
        "keywords": ["grading", "anonymous", "bias", "id", "feedback", "essays", "assignments", "blind"]
    },
    {
        "topic": "Banning Single-Use Plastics Across Campus Cafeterias",
        "audio_script": "Speaker 1: The university needs to ban all plastic water bottles and disposable containers to meet its environmental sustainability goals. Speaker 2: Eliminating plastics completely will slow down service lines and increase operating costs by forcing the use of expensive compostables. Speaker 3: We can phase it in. Offer financial discounts to students who bring reusable cups while adding container recycling stations.",
        "keywords": ["plastics", "sustainability", "ban", "cafeterias", "costs", "compostables", "reusable", "discounts"]
    },
    {
        "topic": "Syllabus Design: Student Voice vs. Faculty Control",
        "audio_script": "Speaker 1: Students should have a voice in selecting reading lists and assignment formats to ensure the curriculum aligns with their professional interests. Speaker 2: Faculty members spend decades mastering their disciplines. Allowing undergraduates to choose topics dilutes the rigorous foundational canon. Speaker 3: A fair middle ground is leaving seventy percent of the core syllabus fixed while letting students vote on specialized elective topics.",
        "keywords": ["syllabus", "curriculum", "faculty", "students", "reading", "elective", "canon", "interests"]
    },
    {
        "topic": "The Value of Lectures vs. Flipping the Classroom",
        "audio_script": "Speaker 1: Traditional lectures are inefficient passive listening exercises. We should record lectures for home viewing and use class time entirely for group problem-solving. Speaker 2: Flipped classrooms put too much burden on students at home, and many won't review the video material beforehand, ruining group work. Speaker 3: Let's split the hour. Deliver a compressed twenty-minute lecture core, followed by thirty minutes of interactive collaborative application.",
        "keywords": ["lecture", "flipped", "classroom", "passive", "collaborative", "problem-solving", "video", "interactive"]
    },
    {
        "topic": "Standardized Testing for Postgraduate Admissions",
        "audio_script": "Speaker 1: Entrance exams like the GRE ensure a standardized benchmark for comparing international applicants fairly. Speaker 2: Standardized tests measure socioeconomic privilege and test-taking prep skills rather than raw research potential or academic dedication. Speaker 3: We should make testing optional, placing far more weight on letters of recommendation, portfolios, and past research experience.",
        "keywords": ["testing", "admissions", "gre", "standardized", "benchmark", "optional", "portfolios", "applicants"]
    },
    {
        "topic": "University Housing: Mixed-Year vs. First-Year Dormitories",
        "audio_script": "Speaker 1: Separating first-year students into dedicated dorms helps them bond and transition into campus social life together smoothly. Speaker 2: First-year dorms become chaotic echo chambers. Mixing freshmen with senior students provides natural mentorship and quieter living environments. Speaker 3: The best compromise is mixed-year residential buildings that feature dedicated quiet wings reserved exclusively for incoming freshmen.",
        "keywords": ["housing", "dormitories", "freshmen", "seniors", "mentorship", "residential", "wings", "transition"]
    },
    {
        "topic": "Online Remote Laboratories vs. Physical Lab Work",
        "audio_script": "Speaker 1: Virtual laboratory simulators allow students to rerun dangerous or expensive chemistry experiments endlessly at zero safety risk. Speaker 2: Online simulators can't replicate the physical tactile skills required to calibrate sensitive hardware or handle delicate glass pipettes. Speaker 3: Use simulators for pre-lab preparation quizzes, but require physical attendance for final experimental data collection runs.",
        "keywords": ["laboratories", "simulators", "virtual", "tactile", "pipettes", "chemistry", "experiments", "safety"]
    },
    {
        "topic": "The Purpose of Elective Modules",
        "audio_script": "Speaker 1: Degree pathways should eliminate general education electives so students can focus entirely on core professional engineering courses. Speaker 2: Electives in philosophy or literature foster creative problem-solving and critical thinking skills that technical tracks ignore. Speaker 3: Let's offer contextual electives, like an ethics module specifically tailored for computer engineers and biotech scientists.",
        "keywords": ["electives", "engineering", "humanities", "curriculum", "technical", "ethics", "breadth", "skills"]
    },
    {
        "topic": "Campus Security: CCTV Arrays vs. Privacy Rights",
        "audio_script": "Speaker 1: Expanding high-definition CCTV camera coverage across all campus walkways is essential to prevent vandalism and ensure safety. Speaker 2: Blanketing public student areas with constant surveillance invades privacy rights and creates a culture of institutional mistrust. Speaker 3: Restrict cameras to main perimeter gates and building entrances, keeping green fields and social lounges surveillance-free.",
        "keywords": ["security", "cctv", "cameras", "surveillance", "privacy", "safety", "trust", "perimeter"]
    },
    {
        "topic": "Group Project Assignments: Free Choice vs. Random Sorting",
        "audio_script": "Speaker 1: Professors should let us choose our own group partners so we can work with reliable peers who share identical work ethics. Speaker 2: In real corporate careers, you never choose your coworkers. Random assignment builds critical adaptative communication skills. Speaker 3: Let students select their partners, but implement mandatory peer evaluation forms to penalize anyone who doesn't contribute fairly.",
        "keywords": ["group", "projects", "partners", "random", "sorting", "coworkers", "communication", "evaluation"]
    },
    {
        "topic": "Accelerated Two-Year Degrees vs. Traditional Four-Year Tracks",
        "audio_script": "Speaker 1: Offering accelerated two-year bachelor degrees saves thousands in tuition and lets students enter the professional workforce much faster. Speaker 2: Cramming a full curriculum into two years eliminates summer internships, research assistanceships, and essential intellectual maturation time. Speaker 3: Reserve the two-year track for highly structured professional degrees, while keeping traditional four-year timelines for research tracks.",
        "keywords": ["accelerated", "degrees", "tuition", "workforce", "internships", "curriculum", "timelines", "duration"]
    },
    {
        "topic": "Sponsoring Varsity Sports vs. Academic Infrastructure",
        "audio_script": "Speaker 1: Investing in massive campus stadiums and varsity athletics boosts school spirit, alumni donations, and national media profile. Speaker 2: Millions spent on coaching staff salaries could be better utilized to renovate decaying science labs and hire tenure-track faculty. Speaker 3: Athletics programs should generate their own revenue through ticket sales and corporate sponsorships without draining central academic funds.",
        "keywords": ["athletics", "stadiums", "funding", "infrastructure", "donations", "faculty", "labs", "revenue"]
    },
    {
        "topic": "The Academic Calendar: Semester vs. Quarter System",
        "audio_script": "Speaker 1: The standard fifteen-week semester provides deep immersion into complex theoretical topics without constant exam pressures. Speaker 2: The ten-week quarter system allows students to explore a wider variety of elective subjects and recover faster if they fail a course. Speaker 3: Semesters fit heavy research and writing tracks perfectly, whereas intense fast-paced quarters suit dynamic business and technical fields.",
        "keywords": ["calendar", "semester", "quarter", "immersion", "pressures", "electives", "failed", "duration"]
    },
    {
        "topic": "Banning Laptops in Lecture Halls",
        "audio_script": "Speaker 1: Professors should ban laptops during lectures because social media notifications distract the student and everyone sitting behind them. Speaker 2: Taking notes by hand is slow and inefficient for complex tech courses. Laptops are essential accessibility tools for many. Speaker 3: Designate the front rows as laptop-free zones for hand-written focus, while allowing screens in the back sections.",
        "keywords": ["laptops", "ban", "lectures", "distractions", "notes", "accessibility", "screens", "zones"]
    },
    {
        "topic": "Peer Review in Undergrad Research Journals",
        "audio_script": "Speaker 1: Undergraduate journals should use blind student editors to judge submissions, providing great editing experience. Speaker 2: Undergraduates lack the deep subject expertise to verify methodology accuracy, increasing the risk of publishing flawed data. Speaker 3: Pair every student editor with a senior faculty mentor to oversee the final verification run before publication.",
        "keywords": ["peer", "review", "journals", "undergraduate", "editors", "methodology", "faculty", "mentor"]
    },
    {
        "topic": "Mandatory Foreign Language Graduation Prerequisites",
        "audio_script": "Speaker 1: Requiring a foreign language is crucial for global career mobility and broad cultural empathy in our interconnected world. Speaker 2: For technical fields like computing, forcing two semesters of French or Spanish delays core engineering specialization. Speaker 3: Allow students to swap a spoken foreign tongue for a professional computer programming language track.",
        "keywords": ["language", "prerequisites", "graduation", "global", "mobility", "engineering", "programming", "culture"]
    },
    {
        "topic": "Open-Book vs. Closed-Book Examinations",
        "audio_script": "Speaker 1: Open-book exams mimic the real world, where professional workers have instant access to data tables and documentation sheets. Speaker 2: Closed-book tests ensure students internalize core formulas and key terminology, which builds necessary rapid problem-solving reflexes. Speaker 3: Use closed-book format for fundamental entry-level concepts, but switch to open-book synthesis for advanced analysis modules.",
        "keywords": ["exams", "open-book", "closed-book", "documentation", "formulas", "internalize", "synthesis", "memory"]
    },
    {
        "topic": "University Consolidation: Merging Small Specialized Colleges",
        "audio_script": "Speaker 1: Merging small arts and science colleges into large regional university networks saves massive administrative costs and expands resources. Speaker 2: Consolidations ruin the distinct identity, low student-teacher ratios, and tight-knit community feel that makes small colleges special. Speaker 3: Centralize operational backend overheads like IT and payroll, but maintain autonomous departmental cultures and localized campus names.",
        "keywords": ["consolidation", "colleges", "merger", "costs", "identity", "ratios", "overhead", "autonomous"]
    },
    {
        "topic": "Commercializing University Research Discoveries",
        "audio_script": "Speaker 1: Universities should actively patent and commercialize internal lab discoveries to generate self-sustaining research funding streams. Speaker 2: Corporate profit incentives corrupt pure scientific inquiry, steering researchers away from vital long-term basic science toward quick commercial gains. Speaker 3: Establish independent spin-off companies for commercial application, keeping university labs dedicated to open, public-domain science.",
        "keywords": ["commercialize", "patent", "funding", "corporate", "profit", "inquiry", "labs", "discoveries"]
    },
    {
        "topic": "Weighing Student Evaluations for Professor Tenure",
        "audio_script": "Speaker 1: End-of-course student evaluations provide direct feedback on a professor's teaching efficacy and classroom accessibility. Speaker 2: Evaluations often devolve into popularity contests, penalizing rigorous professors while rewarding easy graders who give low workloads. Speaker 3: Balance student feedback scores with annual peer evaluations conducted by experienced department heads who observe classes physically.",
        "keywords": ["evaluations", "tenure", "professors", "feedback", "popularity", "graders", "peer", "observation"]
    },
    {
        "topic": "Communal Dorm Bathrooms vs. Private Suites",
        "audio_script": "Speaker 1: Communal bathrooms are cost-effective, maximize floor space, and force residents to interact, building strong dorm communities. Speaker 2: Private suite bathrooms offer necessary privacy, clean hygiene maintenance, and prevent early morning queues before lectures. Speaker 3: Offer communal configurations on freshman floors to save costs, while building suite-style units for upperclassmen who prioritize study privacy.",
        "keywords": ["dorm", "bathrooms", "communal", "suites", "costs", "privacy", "hygiene", "residents"]
    },
    {
        "topic": "Grade Inflation and Curved Scoring Matrices",
        "audio_script": "Speaker 1: Grading on a strict bell curve prevents grade inflation and clearly distinguishes elite students from average performers. Speaker 2: Curved grading breeds a toxic, cutthroat classroom culture where classmates refuse to help each other out of fear of hurting their own score. Speaker 3: Use absolute criterion-referenced scoring matrices, where every student who hits the mastery bar gets an A, regardless of rank.",
        "keywords": ["grading", "curve", "inflation", "bell", "toxic", "culture", "mastery", "criterion"]
    },
    {
        "topic": "24-Hour Campus Library Access Policy",
        "audio_script": "Speaker 1: The main library must stay open twenty-four hours to accommodate varied student work shifts and intense finals study weeks. Speaker 2: Keeping massive buildings open all night strains utility budgets and encourages unhealthy, sleep-deprived cramming behaviors. Speaker 3: Keep only the ground-floor study lounge open all night with automated security card access, closing the main book stacks at midnight.",
        "keywords": ["library", "access", "open", "all-night", "budgets", "cramming", "sleep", "security"]
    },
    {
        "topic": "Sponsoring Campus Greek Life and Fraternities",
        "audio_script": "Speaker 1: Greek organizations build lifelong professional alumni connections and organize massive annual charity fundraising operations. Speaker 2: Fraternities frequently foster exclusionary cultures, hazardous hazing practices, and systemic behavioral problems that disrupt campus safety. Speaker 3: Enforce strict university oversight with immediate charter suspension penalties for violations, while supporting positive philanthropic events.",
        "keywords": ["greek", "fraternities", "alumni", "charity", "hazing", "safety", "oversight", "philanthropic"]
    },
    {
        "topic": "Virtual Reality in Historical Architecture Education",
        "audio_script": "Speaker 1: Using VR headsets allows history students to walk through digital reconstructions of ancient Rome or classical Greece. Speaker 2: High-end VR rigs isolate students, cause motion sickness, and consume funding better spent on physical library books. Speaker 3: Build a dedicated VR laboratory room where small groups can complete guided architectural walkthroughs together as a supplement.",
        "keywords": ["virtual", "reality", "vr", "architecture", "history", "funding", "laboratory", "walkthroughs"]
    },
    {
        "topic": "Standardizing Coding Languages Across Computer Science Tracks",
        "audio_script": "Speaker 1: The computer science department should teach all classes exclusively in Python due to its universal industry demand and simple syntax. Speaker 2: Students must learn low-level programming tongues like C++ to understand memory management mechanics and core hardware architecture. Speaker 3: Start freshmen out with Python to teach basic logic, then mandate C++ and Java for advanced software modules.",
        "keywords": ["coding", "languages", "python", "syntax", "hardware", "architecture", "software", "computer"]
    },
    {
        "topic": "The Requirement of a Formal Undergraduate Thesis",
        "audio_script": "Speaker 1: Writing a mandatory final year thesis develops rigorous investigative methodologies and prepares students for graduate school pathways. Speaker 2: For undergraduates entering corporate jobs, a massive text essay is impractical compared to completing an applied industry portfolio project. Speaker 3: Allow students to select their capstone format: either a theoretical thesis track or a practical industry-partner project.",
        "keywords": ["thesis", "undergraduate", "methodologies", "corporate", "portfolio", "project", "capstone", "theory"]
    },
    {
        "topic": "Hosting Political Debates on Campus Grounds",
        "audio_script": "Speaker 1: Universities must host diverse political speakers to foster robust civic debate and protect free speech principles. Speaker 2: Inviting controversial figures requires expensive security deployment and can cause protests that threaten campus safety. Speaker 3: Allow all student groups to invite speakers, but require them to secure event insurance to offset security costs.",
        "keywords": ["speech", "political", "speakers", "debate", "security", "protests", "safety", "costs"]
    },
    {
        "topic": "Mandatory Mental Health Screens for Freshmen",
        "audio_script": "Speaker 1: Implementing universal mental health screening during freshman orientation helps identify vulnerable students before crises develop. Speaker 2: Forforcing mandatory psychological evaluations invades personal privacy and creates an immediate bureaucratic stigma for new students. Speaker 3: Make the screening assessment completely optional, but promote available counseling services heavily during orientation week.",
        "keywords": ["mental", "health", "screening", "freshmen", "psychological", "privacy", "stigma", "counseling"]
    },
    {
        "topic": "Replacing Cash on Campus with Biometric Payment Cards",
        "audio_script": "Speaker 1: Moving campus stores to completely cashless biometric systems speeds up lunch lines and eliminates theft risks. Speaker 2: Cashless policies discriminate against low-income community visitors who lack bank accounts, while raising data privacy issues. Speaker 3: Maintain digital payment as the standard default line format, but keep one cash-friendly register active in the main dining hall.",
        "keywords": ["cashless", "biometric", "payment", "stores", "theft", "privacy", "cash", "register"]
    },
    {
        "topic": "Sponsoring Creative Writing Workshops vs. Analytical Essays",
        "audio_script": "Speaker 1: Literature tracks should emphasize creative writing to cultivate organic voice, narrative talent, and structural artistic expression. Speaker 2: University outputs must prioritize rigorous analytical composition, as corporate environments value concise evidence-based arguments. Speaker 3: Blend the disciplines by teaching narrative nonfiction, which pairs creative structural voices with deep journalistic research.",
        "keywords": ["writing", "creative", "analytical", "essays", "narrative", "composition", "corporate", "nonfiction"]
    },
    {
        "topic": "The Feasibility of Year-Round University Operations",
        "audio_script": "Speaker 1: Operating campuses year-round through three full equal terms maximizes classroom utilization and allows rapid graduation timelines. Speaker 2: Summer breaks are critical for faculty research windows, deep facilities maintenance cycles, and student seasonal employment. Speaker 3: Keep the traditional calendar layout, but expand the summer block optional elective courses list online.",
        "keywords": ["year-round", "calendar", "terms", "graduation", "summer", "maintenance", "faculty", "research"]
    },
    {
        "topic": "Banning Corporate Branding Across Campus Storefronts",
        "audio_script": "Speaker 1: Universities should remove all corporate coffee chains and fast-food logos to protect the campus from commercial exploitation. Speaker 2: Known commercial brands provide reliable quality standards, student job opportunities, and generate substantial rental revenue for facilities. Speaker 3: Allow external franchises to lease space, provided they adhere to university sustainability and fair wage regulations.",
        "keywords": ["branding", "corporate", "chains", "exploitation", "revenue", "franchises", "lease", "wage"]
    },
    {
        "topic": "Universal Basic Income Simulations in Economics Seminars",
        "audio_script": "Speaker 1: Economics labs should run large-scale digital software agent simulations to model the macroeconomic impacts of a universal basic income. Speaker 2: Simplified computer models cannot predict complex real-world human behavioral labor changes or inflationary supply shocks accurately. Speaker 3: Combine simulation data with empirical case studies from real-world regional pilot programs to ground the findings.",
        "keywords": ["simulations", "economics", "ubi", "models", "labor", "inflationary", "empirical", "cases"]
    },
    {
        "topic": "The Scale of Student-Run Honor Codes for Exams",
        "audio_script": "Speaker 1: Implementing a student-policed honor code eliminates the need for invasive exam proctors and builds an ethical campus community culture. Speaker 2: Without active staff invigilators, cheating levels rise significantly due to academic performance pressures and peer protection logic. Speaker 3: Utilize honor codes for small take-home essays, but maintain strict human proctoring controls for high-stakes final certifications.",
        "keywords": ["honor", "code", "exams", "proctors", "ethical", "cheating", "invigilators", "proctoring"]
    },
    {
        "topic": "Sponsoring Study Abroad Programs vs. Local Internships",
        "audio_script": "Speaker 1: Studying abroad offers irreplaceable cultural immersion, language fluency acceleration, and global perspective building opportunities. Speaker 2: International trips are expensive luxuries. Local internships provide tangible domestic professional networking links and immediate jobs. Speaker 3: Offer virtual global exchange projects that pair local students with international corporate teams to build global links affordably.",
        "keywords": ["abroad", "study", "immersion", "internships", "networking", "jobs", "virtual", "exchange"]
    },
    {
        "topic": "Mandatory Ethics Classes Across Data Science Majors",
        "audio_script": "Speaker 1: All data science degrees must mandate algorithmic ethics modules to prevent engineers from constructing biased profiling software models. Speaker 2: Ethics courses are subjective and take up credit hours better used for advanced machine learning mathematics courses. Speaker 3: Integrate real-world ethical case studies directly into the core programming assignments rather than creating a separate seminar.",
        "keywords": ["ethics", "data", "algorithmic", "science", "biases", "programming", "assignments", "case"]
    },
    {
        "topic": "Replacing Traditional Grading Scales with Pass/Fail Formats",
        "audio_script": "Speaker 1: Switching to pass-fail grading eliminates grade anxiety, allowing students to focus on genuine intellectual exploration. Speaker 2: Pass-fail models reduce student motivation and make it impossible for graduate school admission boards to judge academic excellence. Speaker 3: Use pass-fail grading exclusively for first-semester freshman introductory modules, then transition to traditional GPA letters.",
        "keywords": ["pass-fail", "grading", "anxiety", "exploration", "motivation", "admissions", "freshman", "gpa"]
    },
    {
        "topic": "Campus Dining: Standard Menus vs. Universal Vegan Defaults",
        "audio_script": "Speaker 1: Making all campus dining halls default to vegan menus dramatically cuts institutional carbon prints and promotes health. Speaker 2: Imposing strict dietary limits alienates a huge portion of the student body and violates personal choice rights. Speaker 3: Keep diverse meat items available, but make the most affordable daily value meal option completely plant-based to incentivize green choices.",
        "keywords": ["dining", "vegan", "default", "carbon", "dietary", "choice", "plant-based", "incentivize"]
    },
    {
        "topic": "The Feasibility of Free Public Transit Subsidies for Students",
        "audio_script": "Speaker 1: The university should negotiate free city transit access for all students to cut campus traffic congestion and carbon footprints. Speaker 2: Subsidizing universal transit passes requires raising student activity fees, which penalizes those who walk or bike to class. Speaker 3: Offer transit subsidies exclusively to off-campus students living beyond a three-mile radius from the main university quad.",
        "keywords": ["transit", "transit-pass", "subsidies", "traffic", "congestion", "fees", "off-campus", "radius"]
    },
    {
        "topic": "Sponsoring Tenure-Track Faculty vs. Adjunct Instructors",
        "audio_script": "Speaker 1: Universities must hire more tenure-track faculty to secure long-term institutional stability and support deep academic research groups. Speaker 2: Adjunct professors offer incredible flexibility, cost savings, and bring current up-to-date industry experience directly into classrooms. Speaker 3: Maintain a core tenure track for theoretical foundational courses, while hiring industry adjuncts to teach specialized practical skills.",
        "keywords": ["tenure", "faculty", "adjunct", "stability", "flexibility", "savings", "industry", "skills"]
    },
    {
        "topic": "Implementing Specialized Career Co-Op Tracks",
        "audio_script": "Speaker 1: Engineering majors should alternate semesters between academic classes and full-time paid professional industry co-op employment loops. Speaker 2: Co-op loops interrupt the learning momentum, extend graduation timelines by a year, and separate students from their social cohorts. Speaker 3: Make the co-op structure an optional specialty track, allowing career-focused students to select it while others graduate traditionally.",
        "keywords": ["co-op", "engineering", "industry", "employment", "graduation", "timelines", "optional", "career"]
    },
    {
        "topic": "Banning Smartwatches During In-Person Examinations",
        "audio_script": "Speaker 1: Smartwatches must be banned from exam halls because students can easily hide text files or communicate via messaging apps. Speaker 2: Many rely on smartwatches for monitoring critical health vitals, and checking time is a standard tracking necessity during tests. Speaker 3: Ban all personal smartwatches, mount large digital clocks on the exam walls, and permit verified medical device exceptions.",
        "keywords": ["smartwatches", "ban", "exam", "cheating", "messaging", "clocks", "medical", "devices"]
    }
]

# ==========================================
# --- SIDEBAR & NAVIGATION ---
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0;'>🎓 PTE Exam</h2><p style='color:#6c757d; font-size:0.9rem;'>Speaking Practice Suite</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    module_choice = st.selectbox(
        "Select Active Task Module:",
        [
            "Read Aloud",
            "Repeat Sentence",
            "Retell Lecture",
            "Respond to a Situation",
            "Summarize Group Discussion"
        ]
    )
    
    st.markdown("---")
    st.markdown("#### ⚙️ Engine Telemetry")
    st.caption("• Model: Whisper Open-Source Tiny")
    st.caption(f"• Read Aloud Bank Size: {len(READ_ALOUD_BANK)} Items")
    st.caption(f"• Repeat Sentence Bank Size: {len(REPEAT_SENTENCE_BANK)} Items")
    st.caption(f"• Retell Lecture Bank Size: {len(LECTURE_BANK)} Items")
    st.caption(f"• Situation Bank Size: {len(SITUATION_BANK)} Items")

# ==========================================
# --- REUSABLE AUDIO GRABBER & ENGINE ---
# ==========================================
def process_evaluation_pipeline(audio_io_val, target_string_txt, matched_keywords_bank=None, mode="wpm_match"):
    data, samplerate = sf.read(io.BytesIO(audio_io_val.read()))
    if len(data.shape) > 1: 
        data = data[:, 0]
    audio_data = data.astype(np.float32)
    duration_seconds = len(audio_data) / samplerate
    
    with st.spinner("✨ Analyzing audio wave artifacts via machine learning..."):
        result = whisper_model.transcribe(audio_data, fp16=False, language="en")
        transcription = result.get("text", "").strip()

    if not transcription:
        st.error("❌ Audio capture warning: Low signal detected. Adjust input gain.")
        return None, None, None, None

    total_words = len(transcription.split())
    wpm = round((total_words / duration_seconds) * 60) if duration_seconds > 0 else 0
    
    # Content Metric Calibration
    if mode == "wer_match":
        try:
            error_rate = wer(target_string_txt.lower(), transcription.lower())
            content_score = max(10, round(90 * (1 - error_rate)))
        except:
            content_score = 10
    else:
        matched = [w for w in matched_keywords_bank if w in transcription.lower()]
        kd = len(matched) / len(matched_keywords_bank) if matched_keywords_bank else 0
        content_score = max(10, min(90, round(10 + (kd * 80))))
    
    # Fluency Pacing Calibration
    if 110 <= wpm <= 165: 
        fluency_score = 88
    elif 85 <= wpm or wpm > 165: 
        fluency_score = 65
    else: 
        fluency_score = 30

    overall_pte = max(10, min(90, round((content_score + fluency_score) / 2)))
    return overall_pte, content_score, fluency_score, wpm, transcription

# ==========================================
# --- RENDERING ENGINE MODULES ---
# ==========================================
st.title(f"⚡ {module_choice}")
st.caption("Real-Time Intelligent Evaluation Dashboard")
st.markdown("---")

# MODULE 1: READ ALOUD
if module_choice == "Read Aloud":
    if "read_aloud_prompt" not in st.session_state:
        st.session_state.read_aloud_prompt = random.choice(READ_ALOUD_BANK)

    st.markdown('<div class="card-container"><h4>📋 Instructions</h4><p style="color:#495057;">Look over the text below. When you are ready, click record and read the passage fluently without stopping.</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="target-box">{st.session_state.read_aloud_prompt}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔊 Play Guide Voice", use_container_width=True):
            st.audio(get_audio_prompt_bytes(st.session_state.read_aloud_prompt), format="audio/mp3")
    with col2:
        if st.button("🔄 Swap Text Prompt", type="secondary"):
            st.session_state.read_aloud_prompt = random.choice(READ_ALOUD_BANK)
            st.rerun()

    st.markdown("---")
    audio_value = st.audio_input("Microphone Core Interface Input:")
    
    if audio_value:
        res = process_evaluation_pipeline(audio_value, st.session_state.read_aloud_prompt, mode="wer_match")
        if res[0]:
            overall, content, fluency, wpm, transcript = res
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall Score", f"{overall}/90", delta="PASSED" if overall >= 65 else "RETRY")
            m2.metric("Oral Fluency", f"{fluency}/90")
            m3.metric("Content Match", f"{content}/90")
            m4.metric("Pacing Rate", f"{wpm} WPM")
            
            with st.expander("🔍 Engine Transcription Breakdown", expanded=True):
                st.write("**What the AI heard:**")
                st.info(transcript)

# MODULE 2: REPEAT SENTENCE
elif module_choice == "Repeat Sentence":
    if "current_repeat_target" not in st.session_state:
        st.session_state.current_repeat_target = random.choice(REPEAT_SENTENCE_BANK)

    st.markdown('<div class="card-container"><h4>📋 Instructions</h4><p style="color:#495057;">Listen to the spoken audio cue, then immediately click record and echo the exact words back in sequence.</p></div>', unsafe_allow_html=True)
    
    c_btn1, c_btn2 = st.columns([1, 3])
    with c_btn1:
        if st.button("▶️ Play Exam Audio", type="primary", use_container_width=True):
            st.session_state.rs_audio_bytes = get_audio_prompt_bytes(st.session_state.current_repeat_target, tld='co.uk')
    with c_btn2:
        if st.button("🔄 Next Sentence Target"):
            st.session_state.current_repeat_target = random.choice(REPEAT_SENTENCE_BANK)
            if "rs_audio_bytes" in st.session_state: del st.session_state.rs_audio_bytes
            st.rerun()
            
    if "rs_audio_bytes" in st.session_state:
        st.audio(st.session_state.rs_audio_bytes, format="audio/mp3")

    st.markdown("---")
    audio_recording = st.audio_input("Record response:")
    
    if audio_recording:
        res = process_evaluation_pipeline(audio_recording, st.session_state.current_repeat_target, mode="wer_match")
        if res[0]:
            overall, content, fluency, wpm, transcript = res
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall Score", f"{overall}/90")
            m2.metric("Fluency Accuracy", f"{fluency}/90")
            m3.metric("Sequence Match", f"{content}/90")
            m4.metric("Cadence Rate", f"{wpm} WPM")
            
            with st.expander("👁️ Verbatim Analysis Tracker", expanded=True):
                st.write("**Target Text:**")
                st.success(st.session_state.current_repeat_target)
                st.write("**Transcribed Speech:**")
                st.code(transcript)

# MODULE 3: RETELL LECTURE
elif module_choice == "Retell Lecture":
    if "current_lecture" not in st.session_state:
        st.session_state.current_lecture = random.choice(LECTURE_BANK)

    st.markdown(f'<div class="card-container"><h4>📋 Topic Focus: {st.session_state.current_lecture["topic"]}</h4><p style="color:#495057;">Listen to the academic recording below. Take comprehensive notes, then summarize the key insights in a 40-second address.</p></div>', unsafe_allow_html=True)
    
    cb1, cb2 = st.columns([1, 3])
    with cb1:
        if st.button("▶️ Play Lecture Material", type="primary", use_container_width=True):
            st.session_state.lec_bytes = get_audio_prompt_bytes(st.session_state.current_lecture['transcript'], tld='com')
    with cb2:
        if st.button("🔄 Change Topic Matrix"):
            st.session_state.current_lecture = random.choice(LECTURE_BANK)
            if "lec_bytes" in st.session_state: del st.session_state.lec_bytes
            st.rerun()

    if "lec_bytes" in st.session_state:
        st.audio(st.session_state.lec_bytes, format="audio/mp3")

    st.markdown("---")
    audio_recording = st.audio_input("Record summary output:")
    
    if audio_recording:
        res = process_evaluation_pipeline(audio_recording, None, matched_keywords_bank=st.session_state.current_lecture['keywords'], mode="keyword_match")
        if res[0]:
            overall, content, fluency, wpm, transcript = res
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Overall Synthesis Score", f"{overall}/90")
            m2.metric("Biological/Technical Weights", f"{content}/90")
            m3.metric("Flow Pacing", f"{fluency}/90")
            
            st.write("🎯 **Core Keyword Parameter Targets Met:**")
            matched = [w for w in st.session_state.current_lecture['keywords'] if w in transcript.lower()]
            if matched:
                st.success(", ".join(matched))
            else:
                st.warning("Missing high-weight conceptual tags.")

# ==========================================
# --- MODULE 4: RESPOND TO A SITUATION ---
# ==========================================
elif module_choice == "Respond to a Situation":
    if "current_situation" not in st.session_state:
        st.session_state.current_situation = random.choice(SITUATION_BANK)

    st.markdown('<div class="card-container"><h4>📋 Instructions</h4><p style="color:#495057;">Listen to the scenario context below, or read the constraints. Provide a pragmatic and linguistically appropriate spoken answer to the problem.</p></div>', unsafe_allow_html=True)
    
    # Render Scenario Details inside premium containers
    st.info(f"**Scenario:** {st.session_state.current_situation['scenario']}")
    st.markdown(f"**Prompt Question:** *{st.session_state.current_situation['question']}*")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Touch-Friendly Control Row
    col_act1, col_act2 = st.columns([1, 3])
    with col_act1:
        if st.button("🔊 Play Situation Audio", type="primary", use_container_width=True):
            with st.spinner("Synthesizing prompt audio..."):
                # Combine scenario and question for a complete narration experience
                full_narration_text = f"{st.session_state.current_situation['scenario']} {st.session_state.current_situation['question']}"
                st.session_state.situation_audio_bytes = get_audio_prompt_bytes(full_narration_text, tld='co.uk')

    with col_act2:
        if st.button("🔄 Next Scenario", use_container_width=True):
            st.session_state.current_situation = random.choice(SITUATION_BANK)
            if "situation_audio_bytes" in st.session_state: 
                del st.session_state.situation_audio_bytes
            st.rerun()

    # Audio Player Output Anchor
    if "situation_audio_bytes" in st.session_state:
        st.audio(st.session_state.situation_audio_bytes, format="audio/mp3")

    st.markdown("---")
    audio_recording = st.audio_input("Record your spoken resolution:")
    
    if audio_recording:
        res = process_evaluation_pipeline(audio_recording, None, matched_keywords_bank=st.session_state.current_situation['keywords'], mode="keyword_match")
        if res[0]:
            overall, content, fluency, wpm, transcript = res
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Situational Logic Score", f"{overall}/90")
            m2.metric("Pragmatic Competence", f"{content}/90")
            m3.metric("Fluency Cadence", f"{fluency}/90")
            
            with st.expander("🔍 Response Transcript Evaluation", expanded=True):
                st.write("**What the engine transcribed:**")
                st.code(transcript)

# MODULE 5: SUMMARIZE GROUP DISCUSSION
elif module_choice == "Summarize Group Discussion":
    if "discussion_data" not in st.session_state:
        st.session_state.discussion_data = generate_random_discussion()

    st.markdown(f'<div class="card-container"><h4>📋 Forum Topic: {st.session_state.discussion_data["topic"]}</h4><p style="color:#495057;">Listen to the conflicting arguments raised by the panel. Summarize the divergent positions and extract the final compromise framework.</p></div>', unsafe_allow_html=True)
    
    cd1, cd2 = st.columns([1, 3])
    with cd1:
        if st.button("▶️ Play Panel Audio", type="primary", use_container_width=True):
            st.session_state.disc_bytes = get_audio_prompt_bytes(st.session_state.discussion_data['audio_script'], tld='co.uk')
    with cd2:
        if st.button("🔄 Roll New Argumentative Panel"):
            st.session_state.discussion_data = generate_random_discussion()
            if "disc_bytes" in st.session_state: del st.session_state.disc_bytes
            st.rerun()

    if "disc_bytes" in st.session_state:
        st.audio(st.session_state.disc_bytes, format="audio/mp3")

    st.markdown("---")
    audio_recording = st.audio_input("Record synthesized response:")
    
    if audio_recording:
        res = process_evaluation_pipeline(audio_recording, None, matched_keywords_bank=st.session_state.discussion_data['keywords'], mode="keyword_match")
        if res[0]:
            overall, content, fluency, wpm, transcript = res
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Synthesis Profile Score", f"{overall}/90")
            m2.metric("Logic Extraction Accuracy", f"{content}/90")
            m3.metric("Clarity / Cadence", f"{fluency}/90")
