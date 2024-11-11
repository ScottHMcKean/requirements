# Example usage
def rank_requirements(query: str, requirements: List[str]) -> List[Tuple[str, float]]:
    """Rank requirements by similarity to query"""
    similarity = RequirementSimilarity()
    
    # Calculate similarity scores for each requirement
    ranked_reqs = []
    for req in requirements:
        score, detailed_scores = similarity.get_combined_similarity(query, req)
        ranked_reqs.append((req, score))
    
    # Sort by score in descending order
    ranked_reqs.sort(key=lambda x: x[1], reverse=True)
    return ranked_reqs

# Example
if __name__ == "__main__":
    query = "The system shall provide user authentication via email and password"
    requirements = [
        "Users must be able to log in using their email address and password",
        "The system should implement secure user authentication",
        "Data must be encrypted during transmission",
        "The application should have a responsive design"
    ]
    
    ranked = rank_requirements(query, requirements)
    for req, score in ranked:
        print(f"Score: {score:.3f} - {req}") 