"""
modules/context_builder.py — Phase 2 Semantic Context Engine

This module abstracts raw computer vision detections (Faces + YOLO Objects) 
into high-level, human-readable semantic JSON context.
"""

import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class ContextPayload:
    people: List[str]
    unknown_count: int
    objects: List[str]
    interactions: List[str]
    timestamp: str

class SemanticContextBuilder:
    """
    Transforms raw detection lists into a unified semantic JSON context payload.
    Provides rule-based interaction inference based on co-occurrence of objects.
    """
    def __init__(self):
        # Define basic activity triggers maps (Object -> Interaction)
        self.activity_rules = {
            "cell phone": "using phone",
            "laptop": "working on laptop",
            "book": "reading",
            "bottle": "drinking",
            "cup": "drinking",
            "tv": "watching screen",
            "keyboard": "typing",
            "apple": "eating",
            "sandwich": "eating"
        }

    def build_context(self, 
                      face_names: List[str], 
                      object_labels: List[str], 
                      tz: timezone = timezone.utc) -> str:
        """
        Builds the unified context payload.
        
        Args:
            face_names: List of strings (e.g., ["Viven", "Unknown", "Unknown"])
            object_labels: List of strings (e.g., ["laptop", "cell phone", "chair"])
            tz: Optional timezone for timestamp
            
        Returns:
            JSON formatted string of the context.
        """
        people = []
        unknown_count = 0
        
        # Parse people
        for name in face_names:
            if not name or name.lower() == "unknown":
                unknown_count += 1
            else:
                people.append(name)
                
        # Parse unique objects to avoid duplicate spam (e.g. 5 chairs)
        unique_objects = sorted(list(set(object_labels)))
        
        # Infer interactions
        interactions = self._infer_interactions(len(people) + unknown_count, unique_objects)
        
        payload = ContextPayload(
            people=people,
            unknown_count=unknown_count,
            objects=unique_objects,
            interactions=interactions,
            timestamp=datetime.now(tz).isoformat()
        )
        
        return json.dumps(asdict(payload), indent=2)

    def _infer_interactions(self, total_people: int, objects: List[str]) -> List[str]:
        """Runs heuristic rules to determine what is currently happening."""
        interactions = set()
        
        if total_people == 0:
            return []
            
        # Group dynamic rule
        if total_people > 1:
            interactions.add("group activity")
            
        # Object-based rules
        for obj in objects:
            obj_lower = obj.lower()
            if obj_lower in self.activity_rules:
                interactions.add(self.activity_rules[obj_lower])
                
        return sorted(list(interactions))

# ── Example Usage ────────────────────────────────────────────────
if __name__ == "__main__":
    builder = SemanticContextBuilder()
    
    # Mock data from frame N
    faces = ["Viven", "Unknown"]
    objects = ["laptop", "cell phone", "chair"]
    
    context_json = builder.build_context(faces, objects)
    print("=== JARVIS Context Payload ===")
    print(context_json)
