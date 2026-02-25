"""
Rule-based medical entity extractor using spaCy and medical dictionaries.
"""
import re
import json
from typing import Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


# Medical entity type definitions
MEDICAL_ENTITY_TYPES = {
    "DISEASE": "Medical conditions, diagnoses, illnesses",
    "DRUG": "Medications, pharmaceuticals, drugs",
    "PROCEDURE": "Medical procedures, surgeries, treatments",
    "SYMPTOM": "Patient symptoms, clinical signs",
    "ANATOMY": "Body parts, organs, anatomical structures",
    "TEST": "Diagnostic tests, lab tests, imaging",
    "TREATMENT": "Therapies, interventions"
}


# Common medical entity dictionaries
COMMON_DRUGS = {
    # Pain relievers
    "ibuprofen", "acetaminophen", "aspirin", "naproxen", "diclofenac",
    # Antibiotics
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline", "penicillin",
    "cephalexin", "metronidazole", "trimethoprim", "sulfamethoxazole",
    # Cardiovascular
    "lisinopril", "metoprolol", "amlodipine", "losartan", "atenolol",
    "carvedilol", "warfarin", "atorvastatin", "simvastatin", "rosuvastatin",
    # Diabetes
    "metformin", "glipizide", "insulin", "glimepiride", "sitagliptin",
    # Mental health
    "sertraline", "fluoxetine", "escitalopram", "paroxetine", "duloxetine",
    "venlafaxine", "bupropion", "alprazolam", "lorazepam", "zolpidem",
    # Respiratory
    "albuterol", "fluticasone", "montelukast", "prednisone", "benzonatate",
    # GI
    "omeprazole", "pantoprazole", "esomeprazole", "ranitidine", "famotidine",
    "loperamide", "ondansetron", "metoclopramide",
    # Other common
    "levothyroxine", "prednisone", "methylprednisolone", "hydrochlorothiazide",
    "furosemide", "spironolactone", "allopurinol", "gabapentin", "pregabalin"
}

COMMON_DISEASES = {
    # Cardiovascular
    "hypertension", "heart failure", "atrial fibrillation", "coronary artery disease",
    "myocardial infarction", "arrhythmia", "angina", "stroke", "TIA",
    # Respiratory
    "asthma", "copd", "pneumonia", "bronchitis", "tuberculosis", "lung cancer",
    "pulmonary embolism", "pneumothorax",
    # GI
    "gerd", "gastritis", "peptic ulcer", "crohn's disease", "ulcerative colitis",
    "hepatitis", "cirrhosis", "pancreatitis", "cholecystitis",
    # Endocrine
    "diabetes", "hypothyroidism", "hyperthyroidism", "cushing's syndrome",
    "addison's disease", "polycystic ovary syndrome",
    # Neurological
    "alzheimer's disease", "parkinson's disease", "epilepsy", "migraine",
    "multiple sclerosis", "stroke", "dementia",
    # Mental health
    "depression", "anxiety", "bipolar disorder", "schizophrenia", "ptsd",
    "adhd", "ocd", "panic disorder",
    # Infectious
    "hiv", "aids", "influenza", "covid-19", "malaria", "dengue",
    # Autoimmune
    "rheumatoid arthritis", "lupus", "multiple sclerosis", "psoriasis",
    "inflammatory bowel disease", "celiac disease",
    # Cancer
    "breast cancer", "lung cancer", "colon cancer", "prostate cancer",
    "leukemia", "lymphoma", "melanoma",
    # Other
    "anemia", "arthritis", "osteoporosis", "kidney disease", "liver disease"
}

COMMON_SYMPTOMS = {
    # General
    "fever", "fatigue", "weight loss", "weight gain", "chills", "night sweats",
    # Pain
    "chest pain", "abdominal pain", "headache", "back pain", "joint pain",
    "muscle pain", "nerve pain", "stomach pain",
    # Respiratory
    "cough", "shortness of breath", "wheezing", "sore throat", "congestion",
    "runny nose", "sputum", "hemoptysis",
    # GI
    "nausea", "vomiting", "diarrhea", "constipation", "bloating", "heartburn",
    "dysphagia", "appetite loss", "blood in stool", "melena",
    # Cardiovascular
    "palpitations", "edema", "syncope", "dizziness",
    # Neurological
    "confusion", "seizure", "tremor", "numbness", "tingling", "weakness",
    "vision changes", "hearing loss",
    # Other
    "rash", "itching", "swelling", "urination problems"
}

COMMON_PROCEDURES = {
    # Surgical
    "appendectomy", "cholecystectomy", "colonoscopy", "endoscopy", "laparoscopy",
    "bypass surgery", "angioplasty", "stent placement", "biopsy",
    # Diagnostic
    "mri", "ct scan", "x-ray", "ultrasound", "echocardiogram", "ekg", "ecg",
    "stress test", "catheterization",
    # Therapeutic
    "chemotherapy", "radiation therapy", "dialysis", "intubation",
    "ventilation", "cpr", "defibrillation",
    # Lab
    "blood test", "urinalysis", "biopsy", "culture", "pap smear"
}

COMMON_ANATOMY = {
    # Major organs
    "heart", "lung", "liver", "kidney", "brain", "stomach", "intestine",
    "pancreas", "spleen", "bladder", "thyroid", "prostate", "uterus",
    # Body systems
    "artery", "vein", "nerve", "muscle", "tendon", "ligament", "bone",
    "spine", "skull", "ribcage", "pelvis",
    # Specific
    "coronary artery", "aorta", "vena cava", "pulmonary vein",
    "bronchi", "alveoli", "nephron", "hepatocyte"
}

COMMON_TESTS = {
    # Blood tests
    "complete blood count", "cbc", "basic metabolic panel", "bmp",
    "comprehensive metabolic panel", "cmp", "lipid panel", "liver function test",
    "kidney function test", "thyroid function test", "hba1c", "bnp",
    # Imaging
    "x-ray", "ct scan", "mri", "ultrasound", "mammogram", "pet scan",
    "angiography", "echocardiogram",
    # Other
    "ekg", "ecg", "stress test", "colonoscopy", "endoscopy", "biopsy",
    "urinalysis", "culture", "sensitivity test"
}


@dataclass
class ExtractedEntity:
    """Represents a single extracted medical entity."""
    text: str
    entity_type: str
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    """Result of entity extraction from a text chunk."""
    chunk_id: str
    text: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[tuple[str, str, str]] = field(default_factory=list)


class MedicalEntityExtractor:
    """
    Rule-based medical entity extractor using spaCy and dictionaries.
    Optionally uses an LLM for highly accurate relationship extraction.
    """
    
    def __init__(self, model: str = "en_core_web_md", llm_client: Optional[Any] = None):
        """
        Initialize the medical entity extractor.
        
        Args:
            model: spaCy model to use
            llm_client: Optional LLMClient for hybrid relationship extraction
        """
        self.model_name = model
        self.llm = llm_client
        self._nlp = None
        
        # Compile medical dictionaries
        self._drug_patterns = COMMON_DRUGS
        self._disease_patterns = COMMON_DISEASES
        self._symptom_patterns = COMMON_SYMPTOMS
        self._procedure_patterns = COMMON_PROCEDURES
        self._anatomy_patterns = COMMON_ANATOMY
        self._test_patterns = COMMON_TESTS
        
        # Build regex patterns for each category
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns from dictionaries."""
        # Create case-insensitive patterns
        self._patterns = {
            "DRUG": re.compile(
                r'\b(' + '|'.join(re.escape(drug) for drug in self._drug_patterns) + r')\b',
                re.IGNORECASE
            ),
            "DISEASE": re.compile(
                r'\b(' + '|'.join(re.escape(disease) for disease in self._disease_patterns) + r')\b',
                re.IGNORECASE
            ),
            "SYMPTOM": re.compile(
                r'\b(' + '|'.join(re.escape(symptom) for symptom in self._symptom_patterns) + r')\b',
                re.IGNORECASE
            ),
            "PROCEDURE": re.compile(
                r'\b(' + '|'.join(re.escape(proc) for proc in self._procedure_patterns) + r')\b',
                re.IGNORECASE
            ),
            "ANATOMY": re.compile(
                r'\b(' + '|'.join(re.escape(anat) for anat in self._anatomy_patterns) + r')\b',
                re.IGNORECASE
            ),
            "TEST": re.compile(
                r'\b(' + '|'.join(re.escape(test) for test in self._test_patterns) + r')\b',
                re.IGNORECASE
            ),
        }
        
        # ICD-10 code pattern
        self._icd_pattern = re.compile(
            r'\b[A-Z]\d{2}(?:\.\d{1,4})?\b'
        )
        
        # Drug dosage pattern
        self._dosage_pattern = re.compile(
            r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)\b',
            re.IGNORECASE
        )
    
    def load_model(self):
        """Load the spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self.model_name)
            except OSError:
                print(f"spaCy model '{self.model_name}' not found. Using dictionary-based extraction only.")
                self._nlp = None
    
    def extract(self, text: str, chunk_id: Optional[str] = None) -> ExtractionResult:
        """
        Extract medical entities from text.
        
        Args:
            text: Input text to extract entities from
            chunk_id: Optional chunk identifier
            
        Returns:
            ExtractionResult with extracted entities and relationships
        """
        if not text:
            return ExtractionResult(chunk_id=chunk_id or "", text=text)
        
        entities = []
        
        # Extract using dictionary patterns
        for entity_type, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                entity = ExtractedEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9  # High confidence for dictionary matches
                )
                entities.append(entity)
        
        # Extract ICD-10 codes
        for match in self._icd_pattern.finditer(text):
            entity = ExtractedEntity(
                text=match.group(),
                entity_type="DISEASE",
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.95
            )
            entities.append(entity)
        
        # Use spaCy for additional NER if available
        if self._nlp is not None:
            doc = self._nlp(text)
            
            # Map spaCy entities to medical types
            spacy_mapping = {
                "DISEASE": ["DISEASE", "CONDITION"],
                "CHEMICAL": ["DRUG"],
                "ANATOMY": ["ANATOMY", "BODY_PART"],
            }
            
            for ent in doc.ents:
                # Check if spaCy entity matches our medical types
                mapped_type = None
                for med_type, spacy_types in spacy_mapping.items():
                    if any(spacy_type in ent.label_.upper() for spacy_type in spacy_types):
                        mapped_type = med_type
                        break
                
                if mapped_type:
                    # Only add if not already found
                    if not any(
                        e.text.lower() == ent.text.lower() 
                        for e in entities
                    ):
                        entity = ExtractedEntity(
                            text=ent.text,
                            entity_type=mapped_type,
                            start_pos=ent.start_char,
                            end_pos=ent.end_char,
                            confidence=0.7  # Lower confidence for spaCy NER
                        )
                        entities.append(entity)
        
        # Extract relationships between entities
        relationships = self._extract_relationships(entities, text)
        
        # Deduplicate entities
        unique_entities = self._deduplicate_entities(entities)
        
        return ExtractionResult(
            chunk_id=chunk_id or "",
            text=text,
            entities=unique_entities,
            relationships=relationships
        )
    
    def _deduplicate_entities(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """Remove duplicate entities."""
        seen = set()
        unique = []
        
        for entity in entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        return unique
    
    def _extract_relationships(
        self, 
        entities: list[ExtractedEntity], 
        text: str
    ) -> list[tuple[str, str, str]]:
        """
        Extract relationships between entities based on text patterns or LLM.
        
        Returns:
            List of (source_entity, relationship, target_entity) tuples
        """
        relationships = []
        
        # Build lookup keeping original text capitalization for correct node names
        entities_by_type = defaultdict(list)
        for entity in entities:
            # Avoid duplicates in the list
            if entity.text not in entities_by_type[entity.entity_type]:
                entities_by_type[entity.entity_type].append(entity.text)
                
        # --- LLM-BASED EXTRACTION ---
        if self.llm and len(entities) >= 2:
            entity_list = []
            for e_type, texts in entities_by_type.items():
                for t in texts:
                    entity_list.append(f"- {t} ({e_type})")
            entity_list_str = "\\n".join(entity_list)
            prompt = f"""You are a medical data extractor. I have extracted the following medical entities from a text snippet.
            
Entities:
{entity_list_str}

Text Snippet:
"{text}"

Identify the relationships between only the provided entities based ONLY on the text snippet.
Allowed relationship types:
- TREATS (Drug -> Disease/Symptom)
- CAUSES (Disease -> Symptom/Disease, or Drug -> Symptom)
- HAS_PROCEDURE (Disease -> Procedure)

Output strictly as a JSON list of arrays, e.g.:
[
  ["Lisinopril", "TREATS", "hypertension"],
  ["hypertension", "CAUSES", "headaches"]
]
If there are no clear relationships, output an empty array []. Do not include markdown formatting like ```json.
"""
            try:
                result_str = self.llm.generate([{"role": "user", "content": prompt}], max_tokens=256, temperature=0).strip()
                if result_str.startswith("```json"):
                    result_str = result_str[7:-3]
                elif result_str.startswith("```"):
                    result_str = result_str[3:-3]
                    
                parsed_rels = json.loads(result_str.strip())
                for rel in parsed_rels:
                    if len(rel) == 3:
                        # Verify the entities are actually in our list to avoid LLM hallucinations
                        source, rel_type, target = rel[0], rel[1], rel[2]
                        if any(source.lower() == e.text.lower() for e in entities) and \
                           any(target.lower() == e.text.lower() for e in entities):
                            relationships.append((source, rel_type, target))
                
                return list(set(relationships))
            except Exception as e:
                print(f"LLM extraction failed, falling back to rule-based: {e}")

        # --- RULE-BASED FALLBACK ---
        text_lower = text.lower()
        def check_rel(source_texts, target_texts, rel_type, keywords):
            for source in set(source_texts):
                s_lower = source.lower()
                for target in set(target_texts):
                    t_lower = target.lower()
                    if s_lower == t_lower: continue
                    
                    for kw in keywords:
                        # Find if context looks like "source ... keyword ... target"
                        # Distance constraint ~80 chars to avoid false positives across long sentences
                        pattern = re.escape(s_lower) + r'.{1,80}?' + kw + r'.{1,80}?' + re.escape(t_lower)
                        if re.search(pattern, text_lower):
                            relationships.append((source, rel_type, target))
                            break

        drugs = entities_by_type.get("DRUG", [])
        diseases = entities_by_type.get("DISEASE", [])
        symptoms = entities_by_type.get("SYMPTOM", [])
        procs = entities_by_type.get("PROCEDURE", [])

        # DRUG -[TREATS]-> DISEASE / SYMPTOM
        check_rel(drugs, diseases + symptoms, "TREATS", [
            r'treat', r'manag', r'indicat', r'therap', r'efficac', r'prescrib', r'for'
        ])
        
        # DISEASE -[CAUSES]-> SYMPTOM
        check_rel(diseases, symptoms, "CAUSES", [
            r'caus', r'lead', r'result', r'manifest', r'present', r'symptom'
        ])
        
        # DISEASE -[HAS_PROCEDURE]-> PROCEDURE
        check_rel(diseases, procs, "HAS_PROCEDURE", [
            r'diagnos', r'evaluat', r'surg', r'procedur', r'treat'
        ])
        
        return list(set(relationships))
    
    def extract_batch(
        self, 
        texts: list[tuple[str, str]]
    ) -> list[ExtractionResult]:
        """
        Extract entities from multiple text chunks.
        
        Args:
            texts: List of (chunk_id, text) tuples
            
        Returns:
            List of ExtractionResult objects
        """
        return [
            self.extract(text, chunk_id) 
            for chunk_id, text in texts
        ]


# Default extractor instance
_default_extractor: Optional[MedicalEntityExtractor] = None


def get_entity_extractor(model: str = "en_core_web_md", llm_client: Optional[Any] = None) -> MedicalEntityExtractor:
    """
    Get or create the default entity extractor.
    
    Args:
        model: spaCy model name
        llm_client: Optional LLMClient for relationship extraction
        
    Returns:
        MedicalEntityExtractor instance
    """
    global _default_extractor
    if _default_extractor is None or (_default_extractor.llm is None and llm_client is not None):
        _default_extractor = MedicalEntityExtractor(model, llm_client)
    return _default_extractor
