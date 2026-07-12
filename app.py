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
    # 1. Standardizing Campus Sustainable Infrastructure
    {
        "topic": "Standardizing Campus Sustainable Infrastructure",
        "audio_script": (
            "Speaker 1: If we are truly committed to combating climate change, the university needs to mandate that all new campus construction achieves net-zero emissions. We should immediately phase out all single-use plastics from dining halls, install solar arrays across every rooftop, and retrofit older brick dorms with smart HVAC systems to slash our carbon footprint. It is an ethical obligation to our student body. "
            "Speaker 2: While the sentiment is noble, you are completely ignoring the massive financial reality of these retrofits. Total decarbonization requires tens of millions of dollars in upfront capital. If we sink all our liquid capital into premium green engineering, we will be forced to hike student tuition or cut back on vital departmental research grants. We cannot prioritize environmental optics over the actual affordability of education. "
            "Speaker 3: There is a financially viable middle ground here if we leverage public-private partnerships. The university doesn't have to bear the brunt of the initial costs alone. We can sign long-term power purchase agreements with solar developers who install the panels at zero upfront cost to us, selling us cheaper clean energy over time. Simultaneously, we can implement low-cost, high-impact changes like composting programs and energy-efficient LED lighting upgrades before tackling massive structural overhauls."
        ),
        "keywords": ["sustainability", "infrastructure", "net-zero", "emissions", "tuition", "solar", "retrofits", "capital"]
    },
    
    # 2. Hard Deadlines vs. Flexible Grace Periods
    {
        "topic": "Hard Deadlines vs. Flexible Grace Periods",
        "audio_script": (
            "Speaker 1: Strict assignment deadlines are an outdated, counterproductive relic of rigid bureaucracy. The primary goal of higher education should be demonstrating mastery of the course material, not proving how well someone can manage a crisis under pressure. We need to implement a universal forty-eight-hour grace period for all major assignments to accommodate mental health struggles, sudden illnesses, or unexpected shifts in part-time work schedules. "
            "Speaker 2: I strongly disagree because that approach coddles students and completely fails to prepare them for the realities of the professional workforce. In the corporate sector, a missed deadline can cost a company millions of dollars or break a legally binding contract. If we don't instill time-management skills and accountability right now, we are doing a massive disservice to everyone's career readiness. "
            "Speaker 3: Both perspectives are valid, but we can structure a compromise through a tiered penalty system rather than a black-and-white policy. Let's keep a firm target deadline, but allow students a limited pool of 'tokens' per semester—say, three extension days—that they can use at their own discretion without needing a doctor's note. This maintains structural accountability while offering a realistic safety net for genuine emergencies."
        ),
        "keywords": ["deadlines", "grace-period", "accountability", "workforce", "flexibility", "extensions", "stress", "management"]
    },

    # 3. Micro-Credentials vs. Traditional Four-Year Degrees
    {
        "topic": "Micro-Credentials vs. Traditional Four-Year Degrees",
        "audio_script": (
            "Speaker 1: Traditional four-year degree programs are becoming increasingly obsolete in today's rapidly evolving tech landscape. Students are spending tens of thousands of dollars on broad, filler general education courses when they could instead pursue targeted micro-credentials or specialized digital badges. Industry-recognized certifications in data analytics, cloud architecture, or digital marketing can be completed in six months and lead directly to high-paying jobs. "
            "Speaker 2: That perspective reduces higher education down to mere trade school vocational training. A university education isn't just about learning a specific software tool that will be obsolete in five years; it is about developing deep critical thinking skills, historical context, philosophical grounding, and advanced written communication. Micro-credentials create narrow technicians, whereas a comprehensive degree cultivates well-rounded citizens and leaders. "
            "Speaker 3: We shouldn't view these two pathways as mutually exclusive alternatives. The most forward-thinking solution is to embed micro-credentials directly into the traditional four-year curriculum. For instance, a student majoring in Business Administration or Communication should be able to earn official industry certifications as part of their standard elective coursework. This ensures they graduate with both the philosophical foundation of a degree and the immediate tactical skills demanded by recruiters."
        ),
        "keywords": ["micro-credentials", "degree", "certifications", "vocational", "curriculum", "employability", "skills", "education"]
    },

    # 4. Standardizing Anonymized Grading Protocols
    {
        "topic": "Standardizing Anonymized Grading Protocols",
        "audio_script": (
            "Speaker 1: To completely eliminate implicit bias from academic evaluations, universities must mandate completely anonymized grading across all departments. Professors and teaching assistants are human; they naturally develop positive or negative preconceptions about students based on classroom participation, personality clashes, gender, or race. Blind grading using student ID numbers ensures that an essay or exam is judged purely on its objective merits. "
            "Speaker 2: Anonymizing everything completely destroys the vital relationship between a mentor and a student. Grading isn't just a clinical data-entry task; it is an ongoing formative conversation. If I don't know whose paper I am reading, I cannot tailor my feedback to a specific student's long-term trajectory, acknowledge how far they have progressed since the first week of class, or identify if a normally high-achieving student suddenly needs personal outreach. "
            "Speaker 3: We can resolve this tension by splitting the assessment process into two distinct phases. Initial grading of high-stakes midterms, finals, and standardized essays can be done blindly using student identification numbers to ensure absolute equity and eliminate bias. However, once those baseline scores are locked in, professors should be unmasked so they can add personalized qualitative feedback and adjust final participation marks based on holistic growth."
        ),
        "keywords": ["grading", "anonymized", "bias", "equity", "feedback", "assessment", "evaluation", "blind-grading"]
    },

    # 5. Overhauling the Academic Tenuring System
    {
        "topic": "Overhauling the Academic Tenuring System",
        "audio_script": (
            "Speaker 1: The current academic tenure system has created an insular, unaccountable class of professors who are completely disconnected from teaching quality. Once an academic secures tenure, they have lifetime job security, which often leads to complacency, outdated lectures, and a total lack of responsiveness to student needs. It prevents younger, passionate, and more diverse adjunct instructors from securing stable employment. "
            "Speaker 2: You are fundamentally misunderstanding the entire purpose of tenure. It was never designed to be a reward for complacency; it is a vital shield for academic freedom. Without the absolute protection of tenure, professors could be instantly fired by university administrators or targeted by political special interest groups for conducting controversial research, questioning institutional policies, or teaching challenging historical realities. "
            "Speaker 3: Instead of completely abolishing tenure, we need to introduce rigorous post-tenure review processes every five years. These reviews should hold tenured faculty accountable to modern teaching standards and institutional contributions while firmly preserving their core protection against ideological censorship. If a professor consistently fails their peer evaluations and refuses to update their curriculum, there must be clear remediation frameworks or phased retirement pathways."
        ),
        "keywords": ["tenure", "professors", "academic-freedom", "security", "accountability", "evaluations", "adjunct", "censorship"]
    },

    # 6. Corporate Sponsorship of University Research Labs
    {
        "topic": "Corporate Sponsorship of University Research Labs",
        "audio_script": (
            "Speaker 1: Accepting massive corporate sponsorships for university research labs is a direct threat to scientific integrity. When pharmaceutical, energy, or tech conglomerates fund academic research, they quietly steer the scientific agenda toward commercial viability rather than pure public good. It creates a massive conflict of interest where researchers feel pressured to suppress negative data to secure their next round of funding. "
            "Speaker 2: That is a highly cynical view that ignores the reality of modern research funding. Government grants from organizations like the NSF are fiercely competitive and shrinking every single year. Without private corporate investment, groundbreaking research in cancer therapies, renewable batteries, and quantum computing would completely grind to a halt. These partnerships fast-track innovations from the lab bench directly to real-world deployment. "
            "Speaker 3: The key to resolving this is establishing rigid, non-negotiable firewall policies between the corporate funders and the academic researchers. Corporations should be allowed to provide financial grants, but they must sign ironclad agreements that grant university scientists absolute autonomy over data collection, experimental design, and the unconditional right to publish all findings, regardless of whether the results favor the sponsor's bottom line."
        ),
        "keywords": ["sponsorship", "funding", "corporate", "research", "integrity", "conflict", "grants", "autonomy"]
    },

    # 7. Mandatory General Education Core Curriculums
    {
        "topic": "Mandatory General Education Core Curriculums",
        "audio_script": (
            "Speaker 1: Forcing engineering or computer science majors to take mandatory general education classes in medieval history or poetry is an absolute waste of time and money. Students are paying exorbitant tuition rates to gain specialized technical expertise. Every single credit hour spent on subjects entirely unrelated to their major delays their graduation and adds unnecessary student loan debt. "
            "Speaker 2: I completely disagree with that transactional view of higher education. The ultimate goal of a university is to graduate informed citizens, not highly specialized cogs for the corporate machine. Exposure to the arts, history, and social sciences teaches students how to reason ethically, appreciate diversity, and understand the societal impacts of the technologies they will eventually build. A software engineer who has never studied ethics is a liability. "
            "Speaker 3: The solution lies in redefining how general education is structured. Instead of forcing STEM students into highly abstract, disconnected humanities courses, we should design contextualized core classes. For example, rather than a generic philosophy class, offer a course dedicated specifically to the Ethics of Artificial Intelligence, or a history course focused on the Evolution of Industrial Technology. This bridges the gap effectively."
        ),
        "keywords": ["curriculum", "general-education", "humanities", "stem", "tuition", "ethics", "credits", "interdisciplinary"]
    },

    # 8. Banning vs. Embracing Greek Life Institutions
    {
        "topic": "Banning vs. Embracing Greek Life Institutions",
        "audio_script": (
            "Speaker 1: Fraternities and sororities have become toxic institutions that place an immense liability on modern universities. Year after year, we see systemic issues involving dangerous hazing rituals, substance abuse, exclusionary demographic barriers, and behavioral misconduct. The culture is fundamentally exclusionary and outdated, and the university should take a courageous stand by completely banning Greek life from campus. "
            "Speaker 2: Banning Greek life entirely would destroy some of the most vibrant, supportive, and philanthropic communities on campus. Fraternities and sororities provide a profound sense of belonging, leadership development opportunities, and massive charitable fundraising campaigns for local communities. Furthermore, the alumni networks built within these houses provide students with invaluable career mentoring and job placements after graduation. "
            "Speaker 3: Total abolishment will only drive these groups off-campus and underground, completely removing them from university oversight and making them far more dangerous. The rational approach is to enforce strict institutional regulation. We should mandate live-in adult housing directors, require absolute financial transparency, eliminate pledgeship periods entirely, and hold individual chapters legally and academically accountable for any behavioral infractions."
        ),
        "keywords": ["greek-life", "fraternities", "sororities", "hazing", "philanthropy", "alumni", "regulation", "accountability"]
    },

    # 9. Restructuring the Format of Academic Conferences
    {
        "topic": "Restructuring the Format of Academic Conferences",
        "audio_script": (
            "Speaker 1: The traditional format of academic conferences is incredibly elitist, environmentally destructive, and inefficient. Expecting international researchers to fly thousands of miles, spend thousands of dollars on hotels, and emit tons of carbon just to read a PowerPoint slide for twenty minutes is absurd. We need to permanently transition to fully virtual, open-access online conference models. "
            "Speaker 2: Virtual conferences are an absolute disaster for actual academic collaboration and networking. The real value of a conference doesn't happen during the formal presentations; it happens during the spontaneous hallway conversations, the post-panel coffee breaks, and the casual dinners where collaborative research projects are born. You cannot replicate that human trust and connection through a Zoom screen. "
            "Speaker 3: A permanent hybrid infrastructure is the only logical path forward. We should host localized, regional physical hubs for face-to-face networking, while seamlessly broadcasting all presentations globally via an interactive online platform. This allows junior scholars and researchers from underfunded international institutions to present their work remotely without financial strain, while still preserving physical spaces for collaborative networking."
        ),
        "keywords": ["conferences", "networking", "virtual", "hybrid", "carbon-footprint", "collaboration", "accessibility", "research"]
    },

    # 10. The Legitimacy of Fully Online Degrees
    {
        "topic": "The Legitimacy of Fully Online Degrees",
        "audio_script": (
            "Speaker 1: Fully online degree programs are the greatest democratization of higher education in history. They allow non-traditional students, working parents, and international professionals to access elite university educations without abandoning their careers or uprooting their lives. With modern asynchronous platforms, online degrees are just as rigorous and valid as traditional on-campus variants. "
            "Speaker 2: Online degrees are vastly inferior to the immersive, physical university experience. Sitting alone in a bedroom looking at prerecorded lectures completely strips away the rich, transformative social dynamics of campus life. Online students miss out on spontaneous debates, professor office hours, physical lab experiments, and the campus environment that forces personal growth. It turns education into a transactional commodity. "
            "Speaker 3: The debate shouldn't be about whether online degrees are better or worse, but how we can optimize their delivery. We need to move away from isolated asynchronous models and implement highly synchronous, cohort-based online learning. By utilizing virtual reality lab spaces, mandatory real-time seminar discussions, and regional student meetups, we can capture the community essence of physical campuses while retaining digital flexibility."
        ),
        "keywords": ["online-degrees", "asynchronous", "democratization", "flexibility", "pedagogy", "campus-life", "synchronous", "isolation"]
    },

    # 11. The Role of Standardized Testing in Admissions
    {
        "topic": "The Role of Standardized Testing in Admissions",
        "audio_script": (
            "Speaker 1: Standardized tests like the SAT and GRE are deeply flawed metrics that primarily measure a student's socioeconomic privilege rather than their actual academic potential. Wealthy students can afford expensive private tutoring, test-prep courses, and multiple retakes, which artificially inflates their scores. Universities must adopt permanent test-blind policies to create a truly equitable admissions landscape. "
            "Speaker 2: Abandoning standardized testing entirely forces admissions committees to rely heavily on high school grade point averages, which are notoriously subjective and plagued by rampant grade inflation. A straight-A average from an elite private school means something entirely different than one from an underfunded rural high school. Standardized tests provide the only uniform, objective national benchmark to identify brilliant students from obscure backgrounds. "
            "Speaker 3: The most balanced approach is a holistic, test-optional policy paired with contextual scoring. Admissions offices should evaluate test scores strictly within the context of the student's high school environment. A score of 1300 from a student attending an underfunded school with zero AP classes is arguably more impressive than a 1500 from a student at a top-tier prep academy. Let's use the data as an indicator, not a gatekeeper."
        ),
        "keywords": ["admissions", "testing", "sat", "gre", "equity", "socioeconomic", "holistic", "benchmarks"]
    },

    # 12. Establishing a Universal Campus Minimum Wage
    {
        "topic": "Establishing a Universal Campus Minimum Wage",
        "audio_script": (
            "Speaker 1: The university operates as a multi-million-dollar enterprise and has a moral obligation to pay all student workers a thriving, livable wage. Forcing student researchers, library assistants, and dining staff to work for federal minimum wage while tuition prices continuously skyrocket is exploitative. Establishing a twenty-dollar-an-hour campus minimum wage would dramatically reduce financial anxiety and academic burnout. "
            "Speaker 2: If the university arbitrarily doubles the hourly wage for student workers, the money has to come from somewhere. It will inevitably trigger immediate budget cuts across other vital student services. Departments will be forced to slash the total number of student job openings by half, meaning fewer students will get any on-campus employment opportunities at all. It will inadvertently hurt the very population it aims to help. "
            "Speaker 3: We can offset this financial strain by structuring campus employment as a formal component of financial aid packages through expanded Federal Work-Study subsidies. Additionally, we can prioritize wage increases specifically for high-stress or technical student roles, while offering non-monetary compensation—such as direct housing stipends or meal plan credits—for other positions, ensuring total student compensation increases sustainably."
        ),
        "keywords": ["wages", "minimum-wage", "employment", "budget", "financial-aid", "compensation", "exploitation", "burnout"]
    },

    # 13. Overhauling Student Course Evaluation Metrics
    {
        "topic": "Overhauling Student Course Evaluation Metrics",
        "audio_script": (
            "Speaker 1: End-of-semester student course evaluations are incredibly toxic and statistically useless metrics. Peer-reviewed studies consistently demonstrate that these surveys are heavily influenced by gender bias, racial prejudice, and how leniently a professor grades, rather than actual teaching effectiveness. Universities must stop using these biased popularity contests to determine faculty promotions and raises. "
            "Speaker 2: Students are the primary consumers of higher education, and their voices must matter. If we completely eliminate student evaluations, we remove the only anonymous channel students have to report disorganized professors, hostile classroom environments, or outdated teaching methodologies. Administrators need this direct feedback loop to hold faculty accountable for what actually happens inside the lecture hall. "
            "Speaker 3: The solution is to significantly de-emphasize these subjective end-of-term surveys and replace them with a robust peer-review evaluation model. Faculty teaching quality should be assessed through random, in-person classroom observations by trained pedagogical experts, comprehensive portfolio reviews, and tracking long-term student performance metrics in subsequent, advanced courses. Student surveys should only be used for qualitative feedback."
        ),
        "keywords": ["evaluations", "bias", "faculty", "metrics", "accountability", "peer-review", "pedagogy", "feedback"]
    },

    # 14. Mandating Open Educational Resources (OER)
    {
        "topic": "Mandating Open Educational Resources (OER)",
        "audio_script": (
            "Speaker 1: Commercial textbook publishers are running an absolute racket, charging students hundreds of dollars for minor edition updates that change nothing but the page numbers. The university should officially mandate that all introductory undergraduate courses exclusively utilize Open Educational Resources. Free, peer-reviewed digital textbooks ensure that no student falls behind simply because they cannot afford the reading material. "
            "Speaker 2: A blanket mandate on Open Educational Resources would severely compromise academic freedom and instructional quality. In highly specialized or rapidly evolving fields, open-source textbooks simply do not have the rigorous updates, high-quality interactive software, or extensive test banks that premium publishers provide. Professors must retain absolute authority to select the absolute best educational tools for their classrooms. "
            "Speaker 3: We can build an incentive-based transition program rather than issuing a rigid administrative mandate. The university library can offer competitive grants to professors who take the time to curate, edit, or author high-quality open-source materials for their large-enrollment courses. This honors academic freedom while actively building a robust institutional repository of free textbooks to alleviate the student financial burden."
        ),
        "keywords": ["textbooks", "oer", "open-source", "publishers", "academic-freedom", "grants", "affordability", "curation"]
    },

    # 15. The Value of Lecture Attendance Tracking Systems
    {
        "topic": "The Value of Lecture Attendance Tracking Systems",
        "audio_script": (
            "Speaker 1: Utilizing automated geofencing apps or bluetooth beacons to track student attendance in massive lecture halls is invasive and creepy. It creates a surveillance culture on campus that treats adult university students like untrusted children. If a student decides that reading the lecture slides at home is a better use of their time, they should have the autonomy to make that choice. "
            "Speaker 2: The empirical data on this is absolute: consistent lecture attendance is the single highest predictor of academic success and student retention. When students stop showing up, they isolate themselves and their grades plummet. Attendance tracking apps are not about surveillance; they serve as an early-warning system that allows academic advisors to step in and support struggling students before they completely fail out. "
            "Speaker 3: The technology itself isn't the problem; it is how the data is applied. Instead of tying automated attendance directly to grade penalties, which breeds deep resentment, we should use the tracking data strictly for internal diagnostics. If a student's attendance drops significantly, it should trigger an automated, supportive outreach from counseling or tutoring services to check on their well-being, keeping it helpful rather than punitive."
        ),
        "keywords": ["attendance", "tracking", "surveillance", "autonomy", "retention", "intervention", "data", "privacy"]
    },

    # 16. Quantifying the Value of Legacy Admissions
    {
        "topic": "Quantifying the Value of Legacy Admissions",
        "audio_script": (
            "Speaker 1: Legacy admissions policies—giving preferential treatment to applicants simply because their parents attended the same university—are fundamentally unfair and antithetical to meritocracy. It perpetuates systemic generational privilege and takes away highly competitive enrollment spots from brilliant, first-generation, and marginalized students who earned their way based purely on academic achievement. It is time to ban the practice. "
            "Speaker 2: You are ignoring the massive financial ecosystem that sustains private and public universities alike. Legacy policies foster deep, multi-generational institutional loyalty. This loyalty directly drives the massive alumni donations that fund the university's endowment. Without those critical legacy donations, we wouldn't have the financial capital to build new research facilities or fund generous need-based scholarships for low-income students. "
            "Speaker 3: We can retain alumni engagement without compromising our core admissions ethics by replacing the blunt legacy preference with a philanthropic matching framework. Legacy status shouldn't grant any baseline bonus points in the initial admissions review. Instead, if a legacy applicant meets the rigorous academic criteria entirely on their own merit, their enrollment could trigger a targeted alumni scholarship match for a first-generation student."
        ),
        "keywords": ["legacy", "admissions", "meritocracy", "donations", "endowment", "equity", "scholarships", "privilege"]
    },

    # 17. The Commercialization of College Athletics
    {
        "topic": "The Commercialization of College Athletics",
        "audio_script": (
            "Speaker 1: College athletics have spiraled into an out-of-control commercial enterprise that completely overshadows the academic mission of higher education. Universities are spending hundreds of millions of dollars on luxury stadiums and multi-million-dollar coaches' salaries, while library budgets are slashed and adjunct professors face poverty wages. We need to drastically downscale athletic budgets and return to true amateurism. "
            "Speaker 2: That perspective completely overlooks the fact that major athletic programs operate as massive revenue engines and the primary marketing tool for the entire university. A successful football or basketball team generates immense national media exposure, which leads to a massive surge in freshman applications, increased enrollment revenues, and major corporate partnerships that benefit the entire campus ecosystem, including academics. "
            "Speaker 3: The real issue isn't the money itself, but the lack of institutional equity in how it is distributed. Now that students can profit from their Name, Image, and Likeness, we must mandate that a fixed percentage of all athletic media rights revenue be directly diverted into a central university endowment. This fund can be used exclusively to upgrade academic classrooms, subsidize student housing, and boost faculty salaries."
        ),
        "keywords": ["athletics", "commercialization", "revenue", "marketing", "budgets", "endowment", "nil", "academics"]
    },

    # 18. Centralizing Institutional Review Boards (IRB)
    {
        "topic": "Centralizing Institutional Review Boards (IRB)",
        "audio_script": (
            "Speaker 1: The current campus Institutional Review Board process is a bureaucratic nightmare that stifles scientific progress. Waiting six to nine months for a hyper-conservative committee to review a completely harmless, low-risk survey or ethnographic interview delays critical graduate student research and causes us to lose competitive funding to international labs with faster turnaround times. "
            "Speaker 2: The strict IRB process is the only thin line defending human research subjects from ethical violations and exploitation. History is filled with horrific examples of academic research causing profound psychological or social harm because scientists were moving too fast and lacked objective ethical oversight. The administrative delays are a very small, necessary price to pay for absolute moral accountability and legal protection. "
            "Speaker 3: We can dramatically accelerate the timeline without weakening ethical standards by implementing a centralized, multi-tiered review structure. Low-risk research, such as anonymized public surveys or standard educational observations, should be automatically fast-tracked through an expedited digital review system within forty-eight hours. This allows the full IRB panel to focus their deep scrutiny exclusively on high-risk clinical or vulnerable population studies."
        ),
        "keywords": ["irb", "ethics", "research", "bureaucracy", "expedited", "oversight", "compliance", "protocols"]
    },

    # 19. Banning Laptop Displays in Humanities Seminars
    {
        "topic": "Banning Laptop Displays in Humanities Seminars",
        "audio_script": (
            "Speaker 1: Professors should implement a strict, zero-tolerance ban on laptops and tablets during small humanities seminars. Digital screens create an absolute wall of distraction, with students constantly checking emails, browsing social media, or texting. It completely kills the focused eye contact, deep listening, and spontaneous intellectual engagement that are essential for high-level philosophical discussions. "
            "Speaker 2: A blanket laptop ban is highly discriminatory and ignores modern accessibility needs. Many neurodivergent students, such as those with ADHD or dyslexia, rely heavily on digital devices for real-time note-taking, accessing digital readings, or using assistive screen-reading software. Forcing students to publicly disclose their disabilities just to get a special laptop exemption creates a humiliating classroom environment. "
            "Speaker 3: We can foster an engaged classroom environment without resorting to exclusionary bans by establishing clear 'tech-free engagement zones.' The physical seating layout can be arranged so that the inner circle of the seminar is completely screen-free for active verbal debate, while the outer ring permits device usage for collaborative digital note-taking. This accommodates diverse learning styles while keeping human interaction central."
        ),
        "keywords": ["laptops", "distraction", "accessibility", "neurodiversity", "seminars", "engagement", "pedagogy", "screens"]
    },

    # 20. Shifting to Quantitative Skill-Based Grading
    {
        "topic": "Shifting to Quantitative Skill-Based Grading",
        "audio_script": (
            "Speaker 1: The traditional A-through-F grading system is an arbitrary, opaque metric that fails to communicate what a student actually knows. A student can get a 'C' in a chemistry course due to missed homework assignments, despite completely mastering the actual lab skills. We need to transition entirely to skill-based grading, where transcripts explicitly list specific, verified competencies achieved during the course. "
            "Speaker 2: While skill-based transcripts sound nice in theory, they would create an absolute administrative nightmare for corporate recruiters and graduate school admissions committees. External institutions do not have the time to read through a detailed five-page portfolio of individual micro-competencies for every single applicant. The standard Grade Point Average provides a universally understood, highly efficient shorthand for academic capability. "
            "Speaker 3: The optimal solution is a dual-layered transcript architecture. The front page of the official transcript can maintain the traditional, universally accepted letter grades and cumulative GPA to satisfy external gatekeepers and automated sorting algorithms. However, the reverse side can feature an interactive, verifiable digital portfolio detailing the specific technical skills and competencies mastered, blending efficiency with deep descriptive accuracy."
        ),
        "keywords": ["grading", "transcripts", "competencies", "gpa", "skills", "assessment", "metrics", "credentials"]
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
# Change this inside the Module 5 block:
    if "discussion_data" not in st.session_state:
        st.session_state.discussion_data = random.choice(DISCUSSION_BANK)

    st.markdown(f'<div class="card-container"><h4>📋 Forum Topic: {st.session_state.discussion_data["topic"]}</h4><p style="color:#495057;">Listen to the conflicting arguments raised by the panel. Summarize the divergent positions and extract the final compromise framework.</p></div>', unsafe_allow_html=True)
    
    cd1, cd2 = st.columns([1, 3])
    with cd1:
        if st.button("▶️ Play Panel Audio", type="primary", use_container_width=True):
            st.session_state.disc_bytes = get_audio_prompt_bytes(st.session_state.discussion_data['audio_script'], tld='co.uk')
    with cd2:
        if st.button("🔄 Roll New Argumentative Panel"):
            st.session_state.discussion_data = random.choice(DISCUSSION_BANK)
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
