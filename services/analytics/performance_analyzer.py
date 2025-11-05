"""
Performance Trend Analyzer
Analyzes student performance trends using multiple algorithms:
- Exponential Moving Average (EMA) for smoothing
- Linear Regression for trend analysis
- Anomaly Detection for unusual patterns
- Predictive Analytics for future performance
"""

import numpy as np
from datetime import datetime, timedelta


class PerformanceTrendAnalyzer:
    """Analyzes student performance trends using multiple algorithms"""
    
    def __init__(self, dates, scores):
        """
        Initialize analyzer with performance data
        
        Args:
            dates: List of date strings (e.g., ['Oct 20', 'Oct 21', ...])
            scores: List of performance scores (0-100)
        """
        self.dates = dates if dates else []
        self.scores = np.array(scores) if scores else np.array([])
        
    def analyze(self):
        """
        Complete trend analysis
        
        Returns:
            dict: Complete analysis including smoothed data, trends, predictions, insights
        """
        if len(self.scores) < 2:
            return self._empty_analysis()
        
        # 1. Smooth data with Exponential Moving Average
        smoothed = self._exponential_moving_average()
        
        # 2. Calculate trend with Linear Regression
        trend_line, trend_stats = self._linear_regression()
        
        # 3. Detect anomalies (unusual spikes or dips)
        anomalies = self._detect_anomalies(trend_line)
        
        # 4. Calculate performance metrics
        metrics = self._calculate_metrics()
        
        # 5. Generate AI insights
        insights = self._generate_insights(trend_stats, metrics, anomalies)
        
        return {
            'dates': self.dates,
            'raw_scores': self.scores.tolist(),
            'smoothed_scores': smoothed,
            'trend_line': trend_line,
            'predictions': trend_stats['predictions'],
            'trend_direction': trend_stats['trend'],
            'slope': trend_stats['slope'],
            'metrics': metrics,
            'insights': insights,
            'anomalies': anomalies
        }
    
    def _exponential_moving_average(self, alpha=0.3):
        """
        Calculate Exponential Moving Average for smoothing
        
        Args:
            alpha: Smoothing factor (0-1). Higher = more weight to recent values
            
        Returns:
            list: Smoothed scores
        """
        if len(self.scores) == 0:
            return []
            
        ema = [float(self.scores[0])]
        for i in range(1, len(self.scores)):
            ema_value = alpha * self.scores[i] + (1 - alpha) * ema[i-1]
            ema.append(round(ema_value, 1))
        return ema
    
    def _linear_regression(self):
        """
        Calculate linear regression for trend analysis
        
        Returns:
            tuple: (trend_line, trend_stats)
        """
        x = np.arange(len(self.scores))
        y = self.scores
        n = len(x)
        
        # Calculate slope (m) and intercept (b) for y = mx + b
        x_sum = np.sum(x)
        y_sum = np.sum(y)
        xy_sum = np.sum(x * y)
        x2_sum = np.sum(x**2)
        
        denominator = (n * x2_sum - x_sum**2)
        if denominator == 0:
            # Avoid division by zero
            m = 0
            b = np.mean(y)
        else:
            m = (n * xy_sum - x_sum * y_sum) / denominator
            b = (y_sum - m * x_sum) / n
        
        # Generate trend line
        trend_line = [round(m * i + b, 1) for i in x]
        
        # Predict next 3 days
        predictions = [round(m * (len(x) + i) + b, 1) for i in range(1, 4)]
        # Clamp predictions to 0-100 range
        predictions = [max(0, min(100, p)) for p in predictions]
        
        # Determine trend direction
        if m > 0.5:
            trend = 'improving'
        elif m < -0.5:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return trend_line, {
            'slope': round(m, 2),
            'intercept': round(b, 2),
            'predictions': predictions,
            'trend': trend
        }
    
    def _detect_anomalies(self, trend_line):
        """
        Detect unusual spikes or dips in performance
        
        Args:
            trend_line: Expected values from linear regression
            
        Returns:
            list: Anomalies detected
        """
        anomalies = []
        threshold = 15  # Points deviation from trend
        
        for i, score in enumerate(self.scores):
            deviation = abs(score - trend_line[i])
            if deviation > threshold:
                anomalies.append({
                    'date': self.dates[i],
                    'score': float(score),
                    'expected': trend_line[i],
                    'deviation': round(deviation, 1),
                    'type': 'spike' if score > trend_line[i] else 'dip'
                })
        return anomalies
    
    def _calculate_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
            dict: Performance metrics
        """
        mean = np.mean(self.scores)
        variance = np.var(self.scores)
        std_dev = np.std(self.scores)
        
        # Consistency score (0-100, higher is better)
        # Based on coefficient of variation
        cv = (std_dev / mean * 100) if mean > 0 else 0
        consistency = max(0, 100 - cv)
        
        # Recent momentum (last 5 days vs overall average)
        recent_window = min(5, len(self.scores))
        recent_scores = self.scores[-recent_window:]
        recent_mean = np.mean(recent_scores)
        momentum = recent_mean - mean
        
        return {
            'mean': round(mean, 1),
            'std_dev': round(std_dev, 1),
            'consistency': round(consistency, 1),
            'momentum': round(momentum, 1),
            'min': float(np.min(self.scores)),
            'max': float(np.max(self.scores)),
            'range': float(np.max(self.scores) - np.min(self.scores))
        }
    
    def _generate_insights(self, trend_stats, metrics, anomalies):
        """
        Generate AI insights based on analysis
        
        Args:
            trend_stats: Trend statistics from linear regression
            metrics: Performance metrics
            anomalies: Detected anomalies
            
        Returns:
            list: AI-generated insights
        """
        insights = []
        
        # Trend insight
        slope = abs(trend_stats['slope'])
        if trend_stats['trend'] == 'improving':
            if slope > 3:
                insights.append(f"🚀 Rapid improvement! Gaining {slope:.1f} points/day!")
            elif slope > 1:
                insights.append(f"📈 Steady progress! Improving by {slope:.1f} points/day")
            else:
                insights.append(f"📊 Gradual improvement. Keep up the good work!")
        elif trend_stats['trend'] == 'declining':
            if slope > 3:
                insights.append(f"⚠️ Significant decline ({slope:.1f} points/day). Need support?")
            else:
                insights.append(f"📉 Slight decline. Review recent materials?")
        else:
            if metrics['mean'] > 80:
                insights.append("✨ Excellent stable performance! You've mastered this!")
            else:
                insights.append("📊 Stable performance. Ready for new challenges?")
        
        # Consistency insight
        if metrics['consistency'] > 85:
            insights.append("🎯 Outstanding consistency! Your study routine is excellent!")
        elif metrics['consistency'] > 70:
            insights.append("👍 Good consistency. Keep maintaining your study habits!")
        elif metrics['consistency'] < 50:
            insights.append("⚡ Try more regular study habits for better results")
        
        # Momentum insight
        if metrics['momentum'] > 5:
            insights.append("🔥 Accelerating! You're gaining momentum!")
        elif metrics['momentum'] < -5:
            insights.append("🤔 Recent slowdown detected. Stay focused!")
        
        # Anomaly insight
        if anomalies:
            spikes = sum(1 for a in anomalies if a['type'] == 'spike')
            dips = sum(1 for a in anomalies if a['type'] == 'dip')
            if spikes > 0:
                insights.append(f"⭐ {spikes} exceptional performance{'s' if spikes > 1 else ''} detected!")
            if dips > 0:
                insights.append(f"📊 {dips} unusual dip{'s' if dips > 1 else ''} - review those topics")
        
        # Prediction insight
        if trend_stats['predictions']:
            next_score = trend_stats['predictions'][0]
            if next_score >= 90:
                insights.append(f"🎯 Predicted to reach {next_score}% soon! Almost perfect!")
            elif next_score >= 80:
                insights.append(f"📈 On track to reach {next_score}% - excellent trajectory!")
        
        return insights if insights else ["Complete more assessments to see detailed insights"]
    
    def _empty_analysis(self):
        """
        Return empty analysis for insufficient data
        
        Returns:
            dict: Empty analysis structure
        """
        return {
            'dates': self.dates,
            'raw_scores': [],
            'smoothed_scores': [],
            'trend_line': [],
            'predictions': [],
            'trend_direction': 'unknown',
            'slope': 0,
            'metrics': {
                'mean': 0,
                'std_dev': 0,
                'consistency': 0,
                'momentum': 0,
                'min': 0,
                'max': 0,
                'range': 0
            },
            'insights': ["Complete more assessments to see performance trends"],
            'anomalies': []
        }


# Singleton instance
_analyzer_instance = None

def get_performance_analyzer(dates, scores):
    """
    Get or create performance analyzer instance
    
    Args:
        dates: List of date strings
        scores: List of performance scores
        
    Returns:
        PerformanceTrendAnalyzer: Analyzer instance
    """
    return PerformanceTrendAnalyzer(dates, scores)

