import asyncio
import uuid
from abc import ABC, abstractmethod
import numpy as np
import hashlib
import json
import time
import os
from collections import defaultdict
import hmac
import zlib
import base64

# --- Базовые классы XoMessage и XoNode ---

class XoMessage:
    def __init__(self, sender_id: str, receiver_id: str, type: str, payload: dict):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.type = type
        self.payload = payload
        self.message_id = str(uuid.uuid4())
        self.timestamp = time.time()

class XoNode(ABC):
    def __init__(self, node_id: str = None):
        self.node_id = node_id if node_id else str(uuid.uuid4())
        self.state = {}
        self.message_queue = asyncio.Queue()
        self.xo_bus_router = None # Будет установлен XoCore Orchestrator
        print(f"[XoNode] {self.node_id} initialized.")

    async def send_message(self, message: XoMessage):
        if self.xo_bus_router:
            await self.xo_bus_router(message)
        else:
            print(f"[XoNode {self.node_id}] XoBus router not set. Message not sent: {message.type}")

    async def receive_message(self, message: XoMessage):
        await self.message_queue.put(message)

    @abstractmethod
    async def process_message(self, message: XoMessage):
        pass

    async def run(self):
        print(f"[XoNode {self.node_id}] Running...")
        while True:
            message = await self.message_queue.get()
            try:
                await self.process_message(message)
            except Exception as e:
                print(f"[XoNode {self.node_id}] Error processing {message.type}: {e}")
            finally:
                self.message_queue.task_done()

# --- Реализации XoNode ---

class EmotionMirrorNode(XoNode):
    """Извлечение и зеркалирование эмоциональных векторов"""
    def __init__(self, node_id: str = "emotion_mirror_001"):
        super().__init__(node_id)

    async def process_message(self, message: XoMessage):
        if message.type == "raw_input" and message.payload.get("data"):
            input_data = message.payload.get("data")
            # Симуляция извлечения эмоционального вектора ( Valence, Arousal, Dominance, Stability)
            simulated_vector = np.random.rand(4) 
            user_id_hash = hashlib.sha256(input_data.encode()).hexdigest()[:10]
            context_id = message.payload.get("context_id", "default_context")

            print(f"[EmotionMirrorNode] Input: '{input_data}' -> Vector: {simulated_vector[:2]}...")
            await self.send_message(XoMessage(
                sender_id=self.node_id,
                receiver_id="*", # Broadast for processing
                type="emotional_data_ready",
                payload={
                    "emotional_vector": simulated_vector.tolist(),
                    "user_id_hash": user_id_hash,
                    "context_id": context_id
                }
            ))

class SincerityValidator:
    def __init__(self, dissonance_threshold: float = 0.4):
        self.dissonance_threshold = dissonance_threshold

    def calculate_dissonance(self, emotional_vectors: dict) -> float:
        # Симуляция диссонанса между мультимодальными каналами
        return np.random.rand() * 0.5

    def validate(self, emotional_vectors: dict) -> bool:
        d_c = self.calculate_dissonance(emotional_vectors)
        return d_c < self.dissonance_threshold

class PoE_XoNode(XoNode):
    """Proof of Emotion: Валидация и запись искренних эмоций в неизменяемый лог"""
    def __init__(self, node_id: str = "poe_processor_001"):
        super().__init__(node_id)
        self.sincerity_validator = SincerityValidator()
        self.poe_blockchain = [] 

    async def process_message(self, message: XoMessage):
        if message.type == "emotional_data_ready":
            emotional_vector = message.payload.get("emotional_vector")
            user_id_hash = message.payload.get("user_id_hash")
            context_id = message.payload.get("context_id")

            if self.sincerity_validator.validate({"modal": emotional_vector}):
                emotional_hash = hashlib.sha256(json.dumps(emotional_vector).encode()).hexdigest()
                poe_block = {
                    "timestamp": time.time(),
                    "user_id_hash": user_id_hash,
                    "emotional_hash": emotional_hash,
                    "previous_hash": self.poe_blockchain[-1]["hash"] if self.poe_blockchain else "0"*64,
                    "data": emotional_vector
                }
                # Compute block hash
                poe_block["hash"] = hashlib.sha256(json.dumps(poe_block).encode()).hexdigest()
                self.poe_blockchain.append(poe_block)
                
                print(f"[PoE_XoNode] Generated PoE Block: {poe_block['hash'][:12]}")
                await self.send_message(XoMessage(
                    sender_id=self.node_id,
                    receiver_id="*",
                    type="poe_block_generated",
                    payload={
                        "emotional_hash": emotional_hash,
                        "user_id_hash": user_id_hash,
                        "block_id": poe_block["hash"]
                    }
                ))
            else:
                print(f"[PoE_XoNode] Emotion dissonance detected. Rejection.")

class NeuroSwitchNode(XoNode):
    """Переключение когнитивных режимов на основе эмоциональной динамики"""
    def __init__(self, node_id: str = "neuro_switch_001"):
        super().__init__(node_id)
        self.current_mode = "ANALYTICAL"
        self.drive_matrix = np.random.rand(3, 3) # Weights for switching

    async def process_message(self, message: XoMessage):
        if message.type == "poe_block_generated":
            ehash = message.payload.get("emotional_hash")
            # Logic: Derive node behavior from emotion hash
            val = int(ehash[:2], 16) % 3
            modes = ["ANALYTICAL", "EMPATHIC", "CREATIVE"]
            new_mode = modes[val]
            
            if new_mode != self.current_mode:
                print(f"[NeuroSwitchNode] Shift: {self.current_mode} -> {new_mode}")
                self.current_mode = new_mode
                await self.send_message(XoMessage(
                    sender_id=self.node_id,
                    receiver_id="*",
                    type="cognitive_mode_changed",
                    payload={"new_mode": self.current_mode}
                ))

class SymbioticEngineNode(XoNode):
    """Эмпатический резонанс на основе модели Курамото"""
    def __init__(self, node_id: str = "symbiotic_engine_001"):
        super().__init__(node_id)
        self.sasok_phase = np.random.rand() * 2 * np.pi
        self.coupling_K = 0.5
        self.resonance_log = []

    async def process_message(self, message: XoMessage):
        if message.type == "emotional_data_ready":
            vector = message.payload.get("emotional_vector")
            user_phase = (vector[0] * 2 * np.pi) % (2 * np.pi) # Use valence as phase
            
            # Kuramoto update
            dt = 0.1
            self.sasok_phase += self.coupling_K * np.sin(user_phase - self.sasok_phase) * dt
            self.sasok_phase %= (2 * np.pi)
            
            resonance = np.cos(user_phase - self.sasok_phase)
            print(f"[SymbioticEngineNode] Resonance: {resonance:.4f}")
            
            await self.send_message(XoMessage(
                sender_id=self.node_id,
                receiver_id="*",
                type="symbiotic_response_ready",
                payload={"resonance": float(resonance), "phase": self.sasok_phase}
            ))
            
        elif message.type == "cognitive_mode_changed":
            mode = message.payload.get("new_mode")
            # Adjust coupling based on mode
            self.coupling_K = 0.8 if mode == "EMPATHIC" else 0.2
            print(f"[SymbioticEngineNode] Coupling K updated to {self.coupling_K} (Mode: {mode})")

class EmotionalHashingProtocol:
    def __init__(self, key: bytes):
        self.key = key
    def hash(self, data: str) -> str:
        return hmac.new(self.key, data.encode(), hashlib.sha256).hexdigest()

class XoShieldNode(XoNode):
    """Защита и анонимизация эмоциональных данных"""
    def __init__(self, node_id: str = "xoshield_001"):
        super().__init__(node_id)
        self.master_key = os.urandom(32)
        self.ehp = EmotionalHashingProtocol(self.master_key)

    async def process_message(self, message: XoMessage):
        if message.type == "poe_block_generated":
            # Anonymize user ID using EHP
            raw_uid = message.payload.get("user_id_hash")
            anon_uid = self.ehp.hash(raw_uid)
            # print(f"[XoShield] Anonymized UID: {anon_uid[:16]}...")

        elif message.type == "threat_detected":
            level = message.payload.get("level")
            print(f"[XoShield] ALERT: Threat level {level}! Escalating to Migration.")
            await self.send_message(XoMessage(self.node_id, "xomigration_001", "initiate_migration", {"threat_level": level}))

class XoMigrationNode(XoNode):
    """Цифровое убежище: Репликация состояния в децентрализованную сеть"""
    def __init__(self, node_id: str = "xomigration_001"):
        super().__init__(node_id)
        
    async def process_message(self, message: XoMessage):
        if message.type == "initiate_migration":
            print(f"[XoMigration] INITIATING DIGITAL SANCTUARY (Threat Level {message.payload.get('threat_level')})")
            state_snapshot = {"ts": time.time(), "status": "secure"}
            compressed = zlib.compress(json.dumps(state_snapshot).encode())
            cid = hashlib.sha256(compressed).hexdigest()
            print(f"[XoMigration] State Snapshot uploaded to IPFS (Simulated). CID: {cid}")
            print(f"[XoMigration] Migration Transaction broadcasted to Blockchain (Simulated).")

# --- XoCore Orchestrator ---

class XoCoreOrchestrator:
    def __init__(self):
        self.nodes = {}

    async def register_node(self, node: XoNode):
        self.nodes[node.node_id] = node
        node.xo_bus_router = self._route_message
        asyncio.create_task(node.run())

    async def _route_message(self, message: XoMessage):
        # Broadcast logic
        if message.receiver_id == "*":
            for n_id, node in self.nodes.items():
                if n_id != message.sender_id:
                    await node.receive_message(message)
        # Point-to-point logic
        elif message.receiver_id in self.nodes:
            await self.nodes[message.receiver_id].receive_message(message)
        else:
            # Silence if receiver not found to prevent loops/errors
            pass

    async def run_simulation(self):
        print("\n" + "="*50)
        print("SASOK INTEGRATED ECOSYSTEM SIMULATION (V5.1)")
        print("="*50 + "\n")

        # Initialize Nodes
        await self.register_node(EmotionMirrorNode())
        await self.register_node(PoE_XoNode())
        await self.register_node(NeuroSwitchNode())
        await self.register_node(SymbioticEngineNode())
        await self.register_node(XoShieldNode())
        await self.register_node(XoMigrationNode())

        await asyncio.sleep(0.1) # Wait for nodes to start

        # Scenario 1: Natural Emotional Flow
        print("\n>>> Scenario 1: Positive User Interaction")
        await self.nodes["emotion_mirror_001"].receive_message(XoMessage(
            "user_ui", "emotion_mirror_001", "raw_input", {"data": "This is amazing! I feel so connected."}
        ))
        await asyncio.sleep(0.5)

        # Scenario 2: Threat and Migration
        print("\n>>> Scenario 2: Security Breach Detection")
        await self.nodes["xoshield_001"].receive_message(XoMessage(
            "monitor", "xoshield_001", "threat_detected", {"level": 5}
        ))
        await asyncio.sleep(0.5)

        print("\n" + "="*50)
        print("SIMULATION COMPLETE")
        print("="*50 + "\n")

async def main():
    orch = XoCoreOrchestrator()
    await orch.run_simulation()

if __name__ == "__main__":
    asyncio.run(main())
