from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
from collections import defaultdict
import pandas as pd
import numpy as np

class AnalyticsManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.interactions = []
        self.user_sessions = {}
        self.emotion_data = []
        self.web3_transactions = []

    async def log_interaction(self, interaction_data: Dict) -> Dict:
        """
        Log a user interaction
        """
        try:
            interaction = {
                "id": len(self.interactions) + 1,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": interaction_data.get("user_id"),
                "type": interaction_data.get("type"),
                "data": interaction_data.get("data", {}),
                "session_id": interaction_data.get("session_id")
            }
            
            self.interactions.append(interaction)
            return {"success": True, "interaction_id": interaction["id"]}

        except Exception as e:
            self.logger.error(f"Error logging interaction: {str(e)}")
            raise

    async def log_emotion_data(self, emotion_data: Dict) -> Dict:
        """
        Log emotion analysis results
        """
        try:
            emotion_entry = {
                "id": len(self.emotion_data) + 1,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": emotion_data.get("user_id"),
                "emotion": emotion_data.get("emotion"),
                "confidence": emotion_data.get("confidence"),
                "source": emotion_data.get("source", "text")  # text or image
            }
            
            self.emotion_data.append(emotion_entry)
            return {"success": True, "entry_id": emotion_entry["id"]}

        except Exception as e:
            self.logger.error(f"Error logging emotion data: {str(e)}")
            raise

    async def log_web3_transaction(self, transaction_data: Dict) -> Dict:
        """
        Log Web3 transaction data
        """
        try:
            transaction = {
                "id": len(self.web3_transactions) + 1,
                "timestamp": datetime.utcnow().isoformat(),
                "hash": transaction_data.get("hash"),
                "from_address": transaction_data.get("from"),
                "to_address": transaction_data.get("to"),
                "value": transaction_data.get("value"),
                "status": transaction_data.get("status")
            }
            
            self.web3_transactions.append(transaction)
            return {"success": True, "transaction_id": transaction["id"]}

        except Exception as e:
            self.logger.error(f"Error logging transaction: {str(e)}")
            raise

    async def get_user_statistics(self, user_id: str) -> Dict:
        """
        Get comprehensive statistics for a specific user
        """
        try:
            user_interactions = [i for i in self.interactions if i["user_id"] == user_id]
            user_emotions = [e for e in self.emotion_data if e["user_id"] == user_id]
            user_transactions = [t for t in self.web3_transactions 
                               if t["from_address"] == user_id or t["to_address"] == user_id]

            return {
                "user_id": user_id,
                "total_interactions": len(user_interactions),
                "emotion_distribution": self._calculate_emotion_distribution(user_emotions),
                "transaction_count": len(user_transactions),
                "last_active": max([i["timestamp"] for i in user_interactions]) if user_interactions else None
            }

        except Exception as e:
            self.logger.error(f"Error getting user statistics: {str(e)}")
            raise

    async def get_platform_statistics(self) -> Dict:
        """
        Get overall platform statistics
        """
        try:
            now = datetime.utcnow()
            last_24h = now - timedelta(hours=24)
            
            # Convert ISO timestamps to datetime for comparison
            recent_interactions = [i for i in self.interactions 
                                 if datetime.fromisoformat(i["timestamp"]) > last_24h]

            active_users = len(set(i["user_id"] for i in recent_interactions))
            
            return {
                "total_users": len(set(i["user_id"] for i in self.interactions)),
                "active_users_24h": active_users,
                "total_interactions": len(self.interactions),
                "recent_interactions": len(recent_interactions),
                "emotion_distribution": self._calculate_emotion_distribution(self.emotion_data),
                "transaction_volume": self._calculate_transaction_volume(),
                "timestamp": now.isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting platform statistics: {str(e)}")
            raise

    def _calculate_emotion_distribution(self, emotion_data: List[Dict]) -> Dict:
        """
        Calculate the distribution of emotions from emotion data
        """
        if not emotion_data:
            return {}

        emotion_counts = defaultdict(int)
        for entry in emotion_data:
            emotion_counts[entry["emotion"]] += 1

        total = len(emotion_data)
        return {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}

    def _calculate_transaction_volume(self) -> Dict:
        """
        Calculate transaction volume statistics
        """
        if not self.web3_transactions:
            return {"total": 0, "average": 0}

        values = [float(t["value"]) for t in self.web3_transactions if t["value"]]
        return {
            "total": sum(values),
            "average": sum(values) / len(values) if values else 0
        }

    async def generate_report(self, 
                            start_date: datetime, 
                            end_date: datetime, 
                            report_type: str = "general") -> Dict:
        """
        Generate a comprehensive analytics report
        """
        try:
            # Filter data within date range
            filtered_interactions = [
                i for i in self.interactions 
                if start_date <= datetime.fromisoformat(i["timestamp"]) <= end_date
            ]

            if report_type == "user_engagement":
                report_data = self._generate_user_engagement_report(filtered_interactions)
            elif report_type == "emotion_analysis":
                report_data = self._generate_emotion_analysis_report(
                    [e for e in self.emotion_data 
                     if start_date <= datetime.fromisoformat(e["timestamp"]) <= end_date]
                )
            elif report_type == "web3_activity":
                report_data = self._generate_web3_activity_report(
                    [t for t in self.web3_transactions 
                     if start_date <= datetime.fromisoformat(t["timestamp"]) <= end_date]
                )
            else:
                report_data = self._generate_general_report(filtered_interactions)

            return {
                "report_type": report_type,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "generated_at": datetime.utcnow().isoformat(),
                "data": report_data
            }

        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise

    def _generate_user_engagement_report(self, interactions: List[Dict]) -> Dict:
        """
        Generate user engagement metrics
        """
        df = pd.DataFrame(interactions)
        return {
            "daily_active_users": df.groupby(df["timestamp"].str[:10])["user_id"].nunique().to_dict(),
            "interaction_types": df["type"].value_counts().to_dict(),
            "average_session_duration": 0,  # Placeholder for actual calculation
            "retention_rate": 0  # Placeholder for actual calculation
        }

    def _generate_emotion_analysis_report(self, emotion_data: List[Dict]) -> Dict:
        """
        Generate emotion analysis report
        """
        df = pd.DataFrame(emotion_data)
        return {
            "emotion_trends": df.groupby(df["timestamp"].str[:10])["emotion"].value_counts().to_dict(),
            "average_confidence": df["confidence"].mean(),
            "emotion_distribution": df["emotion"].value_counts(normalize=True).to_dict()
        }

    def _generate_web3_activity_report(self, transactions: List[Dict]) -> Dict:
        """
        Generate Web3 activity report
        """
        df = pd.DataFrame(transactions)
        return {
            "total_volume": df["value"].sum(),
            "average_transaction_value": df["value"].mean(),
            "unique_addresses": len(set(df["from_address"]).union(set(df["to_address"]))),
            "daily_volumes": df.groupby(df["timestamp"].str[:10])["value"].sum().to_dict()
        }

    def _generate_general_report(self, interactions: List[Dict]) -> Dict:
        """
        Generate general platform report
        """
        return {
            "total_interactions": len(interactions),
            "unique_users": len(set(i["user_id"] for i in interactions)),
            "interaction_types": defaultdict(int),
            "peak_usage_times": []  # Placeholder for actual calculation
        }
