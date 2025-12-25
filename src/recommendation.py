"""
Recommendation Engine Module
Provides personalized learning resources based on predicted performance
"""

from __future__ import annotations

from enum import Enum


class PerformanceLevel(Enum):
    """Performance categories for student final grades (0-20 scale)."""
    LOW = "Low Performance"
    AVERAGE = "Average Performance"
    HIGH = "High Performance"


# Thresholds based on Portuguese secondary school grading:
# 0-9: Failing (requires foundational intervention)
# 10-14: Passing to moderate (room for improvement)
# 15-20: Strong to excellent (ready for advanced topics)
PERFORMANCE_THRESHOLDS = {
    "low_upper": 10.0,
    "average_upper": 15.0,
}


def categorize_performance(predicted_score: float) -> PerformanceLevel:
    """
    Categorize predicted final grade into performance level.

    Args:
        predicted_score: Predicted G3 (final grade) on 0-20 scale.

    Returns:
        PerformanceLevel enum representing student performance tier.
    
    Rationale:
        - Low (<10): Below passing threshold; needs foundational support.
        - Average (10-14): Passing; benefits from targeted improvement.
        - High (≥15): Strong performance; ready for enrichment/advanced material.
    """
    if predicted_score < PERFORMANCE_THRESHOLDS["low_upper"]:
        return PerformanceLevel.LOW
    elif predicted_score < PERFORMANCE_THRESHOLDS["average_upper"]:
        return PerformanceLevel.AVERAGE
    else:
        return PerformanceLevel.HIGH


class RecommendationEngine:
    """Generates personalized learning resource recommendations"""
    
    # Learning resources database
    RESOURCES = {
        'low': [
            {
                'type': 'Video Tutorial',
                'title': 'Fundamentals of Mathematics',
                'platform': 'Khan Academy',
                'link': 'https://www.khanacademy.org',
                'difficulty': 'Beginner'
            },
            {
                'type': 'Interactive Course',
                'title': 'Introduction to Problem Solving',
                'platform': 'Codecademy',
                'link': 'https://www.codecademy.com',
                'difficulty': 'Beginner'
            },
            {
                'type': 'Book',
                'title': 'Study Skills Handbook',
                'platform': 'Local Library',
                'difficulty': 'Beginner'
            }
        ],
        'medium': [
            {
                'type': 'Online Course',
                'title': 'Advanced Problem Solving',
                'platform': 'Coursera',
                'link': 'https://www.coursera.org',
                'difficulty': 'Intermediate'
            },
            {
                'type': 'Practice Problems',
                'title': 'Competitive Programming',
                'platform': 'LeetCode',
                'link': 'https://www.leetcode.com',
                'difficulty': 'Intermediate'
            }
        ],
        'high': [
            {
                'type': 'Advanced Course',
                'title': 'Machine Learning Specialization',
                'platform': 'Coursera',
                'link': 'https://www.coursera.org',
                'difficulty': 'Advanced'
            },
            {
                'type': 'Research Paper',
                'title': 'Latest AI & ML Innovations',
                'platform': 'ArXiv',
                'link': 'https://arxiv.org',
                'difficulty': 'Advanced'
            }
        ]
    }
    
    def get_recommendations(self, predicted_score: float) -> dict:
        """
        Get personalized learning resources based on predicted performance.
        
        Args:
            predicted_score: Predicted student performance (0-20 scale).
        
        Returns:
            Dict with performance level, advice, and recommended resources.
        """
        level = categorize_performance(predicted_score)
        
        # Map enum to resource keys
        category_map = {
            PerformanceLevel.LOW: 'low',
            PerformanceLevel.AVERAGE: 'medium',
            PerformanceLevel.HIGH: 'high',
        }
        
        advice_map = {
            PerformanceLevel.LOW: "You need foundational support. Start with basic concepts.",
            PerformanceLevel.AVERAGE: "You're doing well! Focus on advanced topics to improve further.",
            PerformanceLevel.HIGH: "Excellent performance! Explore advanced topics and challenges.",
        }
        
        category = category_map[level]
        
        return {
            'predicted_score': round(predicted_score, 2),
            'performance_level': level.value,
            'advice': advice_map[level],
            'resources': self.RESOURCES[category]
        }
    
    def get_all_resources(self):
        """Get all available resources"""
        return self.RESOURCES


# Example usage
if __name__ == "__main__":
    engine = RecommendationEngine()
    recommendations = engine.get_recommendations(12.5)
    print(f"Recommendations for score 12.5:")
    print(f"Level: {recommendations['performance_level']}")
    print(f"Advice: {recommendations['advice']}")
