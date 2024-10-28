from dataclasses import dataclass, field
from typing import Dict, Set, Any
from collections import defaultdict

@dataclass
class RDFObject:
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)

class RDFStateManager:
    def __init__(self):
        self.objects: Dict[str, RDFObject] = {}
    
    def parse_rdf(self, rdf_text: str) -> Dict[str, RDFObject]:
        """Parse RDF-like text and return dictionary of parsed objects."""
        lines = [line.strip() for line in rdf_text.split('\n') if line.strip()]
        parsed_objects = {}
        
        current_subject = None
        current_object = None
        
        for line in lines:
            if '.' == line:  # End of statement
                current_subject = None
                current_object = None
                continue
                
            # Remove trailing semicolon or period if present
            line = line.rstrip(' ;.')
            parts = line.strip().split(maxsplit=2)
            
            if not parts:  # Skip empty lines
                continue
                
            if ':' in parts[0] and not current_object:  # New subject
                current_subject = parts[0]
                if len(parts) > 1:  # There's more on this line
                    if parts[1] == 'a' and len(parts) > 2:  # Type declaration
                        current_object = RDFObject(type=parts[2])
                        parsed_objects[current_subject] = current_object
            elif parts[0] == 'a' and current_subject:  # Type declaration on new line
                current_object = RDFObject(type=parts[1])
                parsed_objects[current_subject] = current_object
            elif current_object:  # Property line
                predicate = parts[0]
                if len(parts) > 1:
                    object_value = parts[1]
                    # Convert string "true"/"false" to boolean
                    if object_value.lower() == "true":
                        object_value = True
                    elif object_value.lower() == "false":
                        object_value = False
                    current_object.properties[predicate] = object_value
        
        return parsed_objects
    
    def add_or_update(self, rdf_text: str) -> None:
        """Add new objects or update existing ones from RDF text."""
        parsed_objects = self.parse_rdf(rdf_text)
        # Update our state with the parsed objects
        self.objects.update(parsed_objects)
    
    def get_object(self, subject: str) -> RDFObject:
        """Retrieve an object by its subject identifier."""
        return self.objects.get(subject)
    
    def print_state(self) -> None:
        """Print the current state in a readable format."""
        for subject, obj in self.objects.items():
            print(f"{subject}")
            print(f"a {obj.type} ;")
            for pred, value in obj.properties.items():
                print(f"{pred} {format_value(value)} ;")
            print(".")
            print()

def format_value(value: Any) -> str:
    """Format a value for RDF output."""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)