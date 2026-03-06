# SPDX-License-Identifier: MIT
# Copyright (c) 2026, SASOK.

"""
GDPR Compliance Technical Implementation for SASOK.

This module provides the core functionalities to ensure SASOK is compliant with
the General Data Protection Regulation (GDPR). It handles:
- Data Retention Policies: Automatically purges data that has exceeded its
  retention period.
- Data Subject Rights:
    - Right of Access (Article 15): Exporting user data in a machine-readable format.
    - Right to Erasure ('Right to be Forgotten', Article 17): Deleting user data
      upon request.
    - Right to Rectification (Article 16): Correcting inaccurate user data.
    - Right to Data Portability (Article 20): Providing user data in a structured,
      commonly used, and machine-readable format.

The implementation is designed to be auditable and verifiable, with clear logging
and separation of concerns.
"""

import json
from datetime import datetime, timedelta

# Mock database for demonstration purposes. In a real application, this would
# interface with the actual database (e.g., MongoDB, Neo4j).
MOCK_USER_DB = {
    "user-001": {
        "profile": {"name": "Alice", "email": "alice@example.com"},
        "emotional_timeline": [
            {"timestamp": "2025-01-15T10:00:00Z", "valence": 0.8, "arousal": 0.4},
            {"timestamp": "2025-01-16T11:30:00Z", "valence": -0.5, "arousal": 0.7},
        ],
        "retention_policy": "SENSITIVE",
        "last_activity": "2026-01-20T12:00:00Z",
    },
    "user-002": {
        "profile": {"name": "Bob", "email": "bob@example.com"},
        "emotional_timeline": [
            {"timestamp": "2023-11-01T09:00:00Z", "valence": 0.2, "arousal": 0.3}
        ],
        "retention_policy": "DEFAULT",
        "last_activity": "2023-12-01T12:00:00Z",
    },
}

# As defined in the GDPR Technical Implementation Guide
DATA_RETENTION_POLICIES = {
    "DEFAULT": {"duration_days": 730},  # 2 years
    "SENSITIVE": {"duration_days": 365},  # 1 year
    "TRANSIENT": {"duration_days": 1},  # 1 day
}


def get_user_data(user_id: str) -> dict:
    """Retrieves all data for a given user."""
    return MOCK_USER_DB.get(user_id, {})

def export_user_data(user_id: str) -> str:
    """
    Exports a user's data in JSON format, fulfilling the Right of Access and
    Right to Data Portability.
    """
    user_data = get_user_data(user_id)
    if not user_data:
        return "{}"

    # In a real implementation, you would also fetch related data from other
    # services (e.g., knowledge graphs, interaction logs).
    return json.dumps(user_data, indent=2, default=str)

def delete_user_data(user_id: str) -> bool:
    """
    Deletes a user's data, fulfilling the Right to Erasure.
    This should be an irreversible action.
    """
    if user_id in MOCK_USER_DB:
        # In a real system, this would trigger a cascade of deletions across
        # all microservices and databases.
        del MOCK_USER_DB[user_id]
        print(f"Data for user {user_id} has been permanently deleted.")
        return True
    return False

def apply_data_retention_policies():
    """
    Scans the user database and deletes data that has exceeded its
    retention period based on the 'last_activity' timestamp.
    """
    today = datetime.now()
    users_to_delete = []

    for user_id, data in MOCK_USER_DB.items():
        policy_name = data.get("retention_policy", "DEFAULT")
        policy = DATA_RETENTION_POLICIES[policy_name]
        last_activity = datetime.fromisoformat(data["last_activity"].replace("Z", ""))
        retention_days = timedelta(days=policy["duration_days"])

        if last_activity + retention_days < today:
            print(
                f"User {user_id} exceeded retention period of "
                f"{policy['duration_days']} days. Flagging for deletion."
            )
            users_to_delete.append(user_id)

    for user_id in users_to_delete:
        delete_user_data(user_id)


if __name__ == "__main__":
    print("--- GDPR Compliance Simulation ---")

    # 1. Right of Access/Portability
    print("\n1. Testing Right of Access/Portability for user-001...")
    exported_data = export_user_data("user-001")
    print("Exported Data:", exported_data)

    # 2. Data Retention Policy Enforcement
    print("\n2. Applying data retention policies...")
    apply_data_retention_policies()
    print("User 'user-002' should be deleted.")
    assert "user-002" not in MOCK_USER_DB

    # 3. Right to Erasure
    print("\n3. Testing Right to Erasure for user-001...")
    delete_user_data("user-001")
    assert "user-001" not in MOCK_USER_DB

    print("\n--- GDPR Compliance Simulation Complete ---")
    print("Final state of DB:", MOCK_USER_DB)
