# services/knowledge_tracing/models/llm_skill_extractor.py

class LLM_Skill_Extractor:
    """
    Simulates an LLM that performs automated skill extraction from student questions,
    a core component of the SQKT methodology.
    """
    def __init__(self, kcs_map):
        self.kcs_map = kcs_map
        # Create a reverse map from name to ID for easy lookup
        self.name_to_id = {name.lower(): id for id, name in kcs_map.items()}
        print("Initialized LLM Skill Extractor (simulated) for SQKT.")

    def extract_skills_from_question(self, question_text):
        """
        Finds keywords in the question text and maps them to KC IDs.
        A real implementation would use a fine-tuned LLM for semantic analysis.
        """
        extracted_kcs = []
        for name, kc_id in self.name_to_id.items():
            if name in question_text.lower():
                extracted_kcs.append(kc_id)
        return extracted_kcs if extracted_kcs else []
