# services/cognitive_diagnosis/models/explainable_ai_engine.py

class ExplainableAIEngine:
    """
    Simulates an LLM generating explanations for the system's behavior,
    fostering student agency and metacognition.
    """
    def __init__(self, kcs_map):
        self.kcs_map = kcs_map
        print("Explainable AI (XAI) Engine initialized.")

    def explain_recommendation(self, student_graph_obj, action, target_kcs):
        """Generates a justification for the RL agent's chosen action."""
        if not target_kcs:
            return "This exercise will help you practice various programming concepts."
            
        weakest_kc_id = sorted(target_kcs, key=lambda kc: student_graph_obj.graph.edges[student_graph_obj.student_id, kc]['mastery'])[0]
        weakest_kc_name = self.kcs_map[weakest_kc_id]
        mastery_score = student_graph_obj.graph.edges[student_graph_obj.student_id, weakest_kc_id]['mastery']

        explanation = f"Based on your recent activity, the system has identified '{weakest_kc_name}' (current mastery: {mastery_score:.2f}) as a key area for growth. "
        
        if "gen_ex" in action:
            explanation += f"A new, custom problem focusing specifically on this concept will provide targeted practice."
        else:
            explanation += f"The exercise '{action}' is recommended because it directly assesses this skill."
        
        return explanation

    def explain_diagnosis(self, student_id, mastery_profile, reasoning=""):
        """
        Generate explanation for cognitive diagnosis results.
        
        Args:
            student_id (str): Student identifier
            mastery_profile (dict): Current mastery levels for each KC
            reasoning (str): Additional reasoning context
            
        Returns:
            str: Human-readable explanation of the diagnosis
        """
        if not mastery_profile:
            return f"No mastery data available for student {student_id}."
        
        # Find strongest and weakest areas
        sorted_kcs = sorted(mastery_profile.items(), key=lambda x: x[1], reverse=True)
        strongest_kc, strongest_score = sorted_kcs[0]
        weakest_kc, weakest_score = sorted_kcs[-1]
        
        strongest_name = self.kcs_map.get(strongest_kc, strongest_kc)
        weakest_name = self.kcs_map.get(weakest_kc, weakest_kc)
        
        explanation = f"Cognitive Diagnosis for {student_id}:\n\n"
        explanation += f"🎯 Strongest Area: {strongest_name} (mastery: {strongest_score:.2f})\n"
        explanation += f"📈 Growth Opportunity: {weakest_name} (mastery: {weakest_score:.2f})\n\n"
        
        # Categorize mastery levels
        high_mastery = [kc for kc, score in mastery_profile.items() if score >= 0.7]
        medium_mastery = [kc for kc, score in mastery_profile.items() if 0.4 <= score < 0.7]
        low_mastery = [kc for kc, score in mastery_profile.items() if score < 0.4]
        
        if high_mastery:
            explanation += f"✅ Well-mastered concepts ({len(high_mastery)}): "
            explanation += ", ".join([self.kcs_map.get(kc, kc) for kc in high_mastery[:3]])
            if len(high_mastery) > 3:
                explanation += f" and {len(high_mastery) - 3} more"
            explanation += "\n"
        
        if medium_mastery:
            explanation += f"🔄 Developing concepts ({len(medium_mastery)}): "
            explanation += ", ".join([self.kcs_map.get(kc, kc) for kc in medium_mastery[:3]])
            if len(medium_mastery) > 3:
                explanation += f" and {len(medium_mastery) - 3} more"
            explanation += "\n"
        
        if low_mastery:
            explanation += f"🎯 Focus areas ({len(low_mastery)}): "
            explanation += ", ".join([self.kcs_map.get(kc, kc) for kc in low_mastery[:3]])
            if len(low_mastery) > 3:
                explanation += f" and {len(low_mastery) - 3} more"
            explanation += "\n"
        
        if reasoning:
            explanation += f"\n💡 Additional Context: {reasoning}"
        
        return explanation

    def explain_learning_path(self, student_graph_obj, recommended_sequence):
        """
        Generate explanation for recommended learning path.
        
        Args:
            student_graph_obj: Student knowledge graph object
            recommended_sequence (list): Sequence of recommended KCs or exercises
            
        Returns:
            str: Human-readable explanation of the learning path
        """
        if not recommended_sequence:
            return "No specific learning path recommendations at this time."
        
        explanation = f"Personalized Learning Path for {student_graph_obj.student_id}:\n\n"
        
        for i, item in enumerate(recommended_sequence[:5], 1):  # Show first 5 items
            if item in self.kcs_map:
                kc_name = self.kcs_map[item]
                current_mastery = student_graph_obj.get_mastery_profile().get(item, 0.0)
                explanation += f"{i}. {kc_name} (current mastery: {current_mastery:.2f})\n"
            else:
                explanation += f"{i}. {item}\n"
        
        if len(recommended_sequence) > 5:
            explanation += f"... and {len(recommended_sequence) - 5} more steps\n"
        
        explanation += "\n🎯 This sequence is optimized based on your current knowledge state and learning prerequisites."
        
        return explanation
